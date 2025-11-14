// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/framework/resource_management/execution_manager.h"

#include <atomic>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/sampler.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/core/tasks.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/executor_data_util.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {

absl::StatusOr<TaskId> ExecutionManager::GetNewTaskId() {
  return next_task_id_.fetch_add(1);
}

absl::Status ExecutionManager::CreateTask(
    TaskId task_id,
    absl::AnyInvocable<absl::StatusOr<Responses>(
        absl::AnyInvocable<void(absl::StatusOr<Responses>)>&
            callback)> absl_nonnull task,
    absl::flat_hash_set<TaskId> dependent_tasks,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
  absl::MutexLock lock(task_lookup_mutex_);
  if (task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " already exists in task list."));
  }
  TaskState task_state = TaskState::kCreated;
  for (TaskId dep_task_id : dependent_tasks) {
    if (!task_lookup_.contains(dep_task_id)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Dependency task ", dep_task_id, " not found in task list."));
    }
    if (task_lookup_.at(dep_task_id).task_state == TaskState::kFailed ||
        task_lookup_.at(dep_task_id).task_state ==
            TaskState::kDependentTaskFailed) {
      task_state = TaskState::kDependentTaskFailed;
    } else if (task_lookup_.at(dep_task_id).task_state != TaskState::kDone) {
      task_lookup_.at(dep_task_id).following_tasks.insert(task_id);
    }
  }
  TaskInfo task_info;
  task_info.task_state = task_state;
  task_info.task = std::move(task);
  task_info.dependent_tasks = std::move(dependent_tasks);
  task_info.callback = std::move(callback);
  task_lookup_.insert({task_id, std::move(task_info)});

  // Signal task state update with callback.
  task_lookup_.at(task_id).callback(Responses(task_state));

  // If there are no dependency tasks, we can queue the task immediately.
  // Otherwise, the task will be queued when all dependency tasks are done.
  if (task_lookup_.at(task_id).dependent_tasks.empty()) {
    RETURN_IF_ERROR(QueueTask(task_id));
  }
  return absl::OkStatus();
}

absl::Status ExecutionManager::QueueTask(TaskId task_id) {
  RETURN_IF_ERROR(
      ConfirmTaskState(task_id, TaskState::kCreated));
  if (!task_lookup_.at(task_id).dependent_tasks.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " has dependent tasks not finished."));
  }

  RETURN_IF_ERROR(execution_thread_pool_->Schedule([this, task_id]() {
    ABSL_LOG(INFO) << "Executing task with task id: " << task_id;
    auto task_callback_pair = StartTask(task_id);
    if (!task_callback_pair.ok()) {
      ABSL_LOG(INFO) << "Failed to start task: " << task_callback_pair.status()
                     << " with task id: " << task_id;
      return;
    }
    auto responses =
        task_callback_pair.value().first(task_callback_pair.value().second);
    auto status = FinishTask(task_id, std::move(responses),
                             std::move(task_callback_pair.value().second));
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to finish task: " << status
                      << " with task id: " << task_id;
      return;
    }
  }));

  task_lookup_.at(task_id).task_state = TaskState::kQueued;
  task_lookup_.at(task_id).callback(
      Responses(task_lookup_.at(task_id).task_state));

  return absl::OkStatus();
}

absl::StatusOr<std::pair<
    absl::AnyInvocable<absl::StatusOr<Responses>(
        absl::AnyInvocable<void(absl::StatusOr<Responses>)>& callback)>,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
ExecutionManager::StartTask(TaskId task_id) {
  absl::MutexLock lock(task_lookup_mutex_);
  RETURN_IF_ERROR(ConfirmTaskState(task_id, TaskState::kQueued));
  if (task_lookup_.at(task_id).task == nullptr) {
    auto error_status =
        absl::InvalidArgumentError("Task is null when trying to start.");
    task_lookup_.at(task_id).callback(error_status);
    return error_status;
  }
  task_lookup_.at(task_id).task_state = TaskState::kProcessing;
  task_lookup_.at(task_id).callback(
      Responses(task_lookup_.at(task_id).task_state));
  return std::make_pair(std::move(task_lookup_.at(task_id).task),
                        std::move(task_lookup_.at(task_id).callback));
}

absl::Status ExecutionManager::FinishTask(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (!callback) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " has null callback."));
  }
  auto invoke_callback_and_return = [&](absl::Status status) -> absl::Status {
    callback(status);
    return status;
  };
  {
    absl::MutexLock lock(task_lookup_mutex_);
    auto status = ConfirmTaskState(task_id, TaskState::kProcessing,
                                   /*trigger_callback=*/false);
    if (!status.ok()) {
      return invoke_callback_and_return(status);
    }
    if (!responses.ok()) {
      auto following_waiting_tasks = FollowingWaitingTasks(task_id);
      if (!following_waiting_tasks.ok()) {
        return invoke_callback_and_return(following_waiting_tasks.status());
      }
      for (TaskId following_task_id : following_waiting_tasks.value()) {
        task_lookup_.at(following_task_id).dependent_tasks.clear();
        if (task_lookup_.at(following_task_id).callback) {
          task_lookup_.at(following_task_id)
              .callback(Responses(TaskState::kDependentTaskFailed));
        }
        task_lookup_.at(following_task_id).task_state =
            TaskState::kDependentTaskFailed;
      }
      callback(responses.status());
      task_lookup_.at(task_id).task_state = TaskState::kFailed;
    } else {
      for (TaskId following_task_id :
           task_lookup_.at(task_id).following_tasks) {
        auto following_task_status = ConfirmTaskState(
            following_task_id, TaskState::kCreated);
        if (!following_task_status.ok()) {
          return invoke_callback_and_return(following_task_status);
        }
        if (!task_lookup_.at(following_task_id)
                 .dependent_tasks.contains(task_id)) {
          auto error_status = absl::InvalidArgumentError(
              absl::StrCat("Following task ", following_task_id,
                           " does not depend on task ", task_id));
          return invoke_callback_and_return(error_status);
        }
        task_lookup_.at(following_task_id).dependent_tasks.erase(task_id);
        if (task_lookup_.at(following_task_id).dependent_tasks.empty()) {
          RETURN_IF_ERROR(QueueTask(following_task_id));
        }
      }
      auto task_state = responses->GetTaskState();
      callback(std::move(responses));
      task_lookup_.at(task_id).task_state = task_state;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::flat_hash_set<TaskId>>
ExecutionManager::FollowingWaitingTasks(TaskId task_id) {
  absl::flat_hash_set<TaskId> following_waiting_tasks;
  for (TaskId following_task_id : task_lookup_.at(task_id).following_tasks) {
    if (!task_lookup_.contains(following_task_id)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Following task ", following_task_id, " not found in task list."));
    }
    if (!task_lookup_.at(following_task_id).dependent_tasks.contains(task_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Following task ", following_task_id,
                       " does not depend on task ", task_id));
    }
    if (task_lookup_.at(following_task_id).task_state ==
        TaskState::kCreated) {
      following_waiting_tasks.insert(following_task_id);
      ASSIGN_OR_RETURN(auto next_following_waiting_tasks,
                       FollowingWaitingTasks(following_task_id));
      following_waiting_tasks.insert(next_following_waiting_tasks.begin(),
                                     next_following_waiting_tasks.end());
    }
  }
  return following_waiting_tasks;
}

absl::Status ExecutionManager::ConfirmTaskState(TaskId task_id,
                                                TaskState expected_state,
                                                bool trigger_callback) {
  if (!task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " not found in task list."));
  }
  if (trigger_callback && task_lookup_.at(task_id).callback == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " has null callback."));
  }
  if (task_lookup_.at(task_id).task_state != expected_state) {
    auto error_status = absl::InvalidArgumentError(absl::StrCat(
        "Task ", task_id, " is not in ", expected_state, " task_state."));
    if (trigger_callback) {
      task_lookup_.at(task_id).callback(error_status);
    }
    return error_status;
  }
  return absl::OkStatus();
}

absl::StatusOr<ExecutorInputs> ExecutionManager::ProcessAndCombineContents(
    const std::vector<InputData>& preprocessed_contents) {
  std::vector<int> combined_token_ids;
  std::vector<ExecutorVisionData> all_image_data;
  std::vector<ExecutorAudioData> all_audio_data;
  for (const auto& preprocessed_content : preprocessed_contents) {
    if (const auto* input_text =
            std::get_if<InputText>(&preprocessed_content)) {
      ASSIGN_OR_RETURN(const auto* token_ids,
                       input_text->GetPreprocessedTextTensor());
      if (token_ids == nullptr) {
        return absl::InvalidArgumentError(
            "Token IDs is null in preprocessed_contents.");
      }
      LITERT_ASSIGN_OR_RETURN(auto ids_buffer_span,
                              ReferTensorBufferAsSpan<int>(*token_ids));
      combined_token_ids.insert(combined_token_ids.end(),
                                ids_buffer_span.begin(), ids_buffer_span.end());
    } else if (const auto* input_image =
                   std::get_if<InputImage>(&preprocessed_content)) {
      ASSIGN_OR_RETURN(const auto* image_tensor,
                       input_image->GetPreprocessedImageTensor());
      if (image_tensor == nullptr) {
        return absl::InvalidArgumentError(
            "Image tensor is null in preprocessed_contents.");
      }
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("vision_executor"));
      }
      ASSIGN_OR_RETURN(auto single_image_data,
                       vision_executor_->Encode(*image_tensor));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("vision_executor"));
      }
      ASSIGN_OR_RETURN(auto embeddings_ptr,
                       single_image_data.GetEmbeddingsPtr());
      const auto& dimensions = TensorBufferDims(*embeddings_ptr);
      // The last two dimensions are [..., image_token_num, model_dimension].
      const int image_token_num = dimensions.at(dimensions.size() - 2);
      combined_token_ids.insert(combined_token_ids.end(), image_token_num,
                                ExecutorVisionData::kSpecialToken);
      all_image_data.push_back(std::move(single_image_data));
    } else if (const auto* input_audio =
                   std::get_if<InputAudio>(&preprocessed_content)) {
      ASSIGN_OR_RETURN(const auto* spectrogram_tensor,
                       input_audio->GetPreprocessedAudioTensor());
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("audio_executor"));
      }
      ASSIGN_OR_RETURN(auto single_audio_data,
                       audio_executor_->Encode(*spectrogram_tensor));
      if (benchmark_info_.has_value()) {
        RETURN_IF_ERROR(benchmark_info_->TimeMarkDelta("audio_executor"));
      }
      const int num_audio_tokens = single_audio_data.GetValidTokens();
      all_audio_data.push_back(std::move(single_audio_data));
      combined_token_ids.insert(combined_token_ids.end(), num_audio_tokens,
                                ExecutorAudioData::kSpecialToken);
      combined_token_ids.push_back(ExecutorAudioData::kEndToken);
    }
  }

  if (combined_token_ids.empty()) {
    return absl::InvalidArgumentError(
        "No token IDs found in preprocessed_contents.");
  }

  std::optional<ExecutorVisionData> combined_image_data = std::nullopt;
  if (!all_image_data.empty()) {
    ASSIGN_OR_RETURN(combined_image_data,
                     CombineExecutorVisionData(all_image_data));
  }
  std::optional<ExecutorAudioData> combined_audio_data = std::nullopt;
  if (!all_audio_data.empty()) {
    ASSIGN_OR_RETURN(combined_audio_data,
                     CombineExecutorAudioData(all_audio_data));
  }

  last_prefill_token_id_ = combined_token_ids.back();

  ASSIGN_OR_RETURN(auto token_ids_buffer,
                   tokenizer_->TokenIdsToTensorBuffer(combined_token_ids));

  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::move(combined_image_data),
                        std::move(combined_audio_data));
  return inputs;
}

absl::StatusOr<std::unique_ptr<ExecutionManager>> ExecutionManager::Create(
    Tokenizer* absl_nonnull tokenizer,
    std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
    std::unique_ptr<VisionExecutor> vision_executor,
    std::unique_ptr<AudioExecutor> audio_executor,
    std::unique_ptr<Sampler> sampler, SessionConfig session_config) {
  return absl::WrapUnique(new ExecutionManager(
      tokenizer, std::move(llm_executor), std::move(vision_executor),
      std::move(audio_executor), std::move(sampler), session_config,
      /*benchmark_info=*/std::nullopt));
}

absl::Status ExecutionManager::WaitUntilDone(TaskId task_id,
                                             absl::Duration timeout) {
  auto task_done = [this, task_id]() {
    task_lookup_mutex_.AssertReaderHeld();
    return task_lookup_.contains(task_id) &&
           (task_lookup_.at(task_id).task_state == TaskState::kDone ||
            task_lookup_.at(task_id).task_state == TaskState::kFailed ||
            task_lookup_.at(task_id).task_state ==
                TaskState::kDependentTaskFailed);
  };
  absl::MutexLock lock(task_lookup_mutex_);
  return task_lookup_mutex_.AwaitWithTimeout(absl::Condition(&task_done),
                                             timeout)
             ? absl::OkStatus()
             : absl::DeadlineExceededError(absl::StrCat(
                   "Task ", task_id, " did not complete within the timeout of ",
                   absl::FormatDuration(timeout), "."));
}

absl::Status ExecutionManager::WaitUntilAllDone(absl::Duration timeout) {
  return execution_thread_pool_->WaitUntilDone(timeout);
}

absl::Status ExecutionManager::AddPrefillTask(
    TaskId task_id, std::vector<InputData> inputs,
    absl::flat_hash_set<TaskId> dep_tasks,
    std::optional<BenchmarkInfo>& benchmark_info,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }
  if (inputs.size() != 1) {
    callback(absl::InvalidArgumentError(
        absl::StrCat("Prefill task expects 1 input, but got ", inputs.size())));
    return absl::InvalidArgumentError(
        absl::StrCat("Prefill task expects 1 input, but got ", inputs.size()));
  }

  absl::AnyInvocable<absl::StatusOr<Responses>(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> & callback)>
      task = [this, inputs = std::move(inputs), &benchmark_info](
                 absl::AnyInvocable<void(absl::StatusOr<Responses>)>&
                     callback) mutable -> absl::StatusOr<Responses> {
    ASSIGN_OR_RETURN(auto executor_inputs, ProcessAndCombineContents(inputs));

    return Tasks::Prefill(*llm_executor_.get(), executor_inputs,
                          /*wait_for_completion=*/true,
                          /*benchmark_info=*/benchmark_info);
  };

  return CreateTask(task_id, std::move(task), std::move(dep_tasks),
                    std::move(callback));
}

absl::Status ExecutionManager::AddDecodeTask(
    TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    int num_output_candidates, Constraint* absl_nullable constraint,
    std::shared_ptr<std::atomic<bool>> cancelled,
    std::optional<BenchmarkInfo>& benchmark_info,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  absl::AnyInvocable<absl::StatusOr<Responses>(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> & callback)>
      task =
          [this, num_output_candidates, constraint, cancelled,
           &benchmark_info](absl::AnyInvocable<void(absl::StatusOr<Responses>)>&
                                callback) mutable -> absl::StatusOr<Responses> {
    // TODO(hoko) grab benchmark info from execution manager.
    stop_token_detector_->ResetBatch(num_output_candidates);
    std::optional<Sampler*> sampler = std::nullopt;
    std::optional<litert::TensorBuffer> decoded_ids_buffer = std::nullopt;
    if (sampler_ != nullptr) {
      sampler = sampler_.get();
      std::vector<int> decoded_ids(num_output_candidates,
                                   last_prefill_token_id_);
      LITERT_ASSIGN_OR_RETURN(
          decoded_ids_buffer,
          CopyToTensorBuffer<int>(decoded_ids, {num_output_candidates, 1}));
    }
    return Tasks::Decode(
        *llm_executor_.get(), *tokenizer_, *stop_token_detector_,
        num_output_candidates, benchmark_info, sampler, constraint,
        std::move(decoded_ids_buffer), callback, cancelled.get());
  };

  return CreateTask(task_id, std::move(task), std::move(dep_tasks),
                    std::move(callback));
}

}  // namespace litert::lm
