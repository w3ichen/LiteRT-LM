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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_H_

#include <atomic>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/sampler.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor.h"
#include "runtime/framework/threadpool.h"

namespace litert::lm {

using TaskId = int;

// All the information about a task.
// - task: The task function. This is the function that will be executed by the
//   execution manager. Will be retrieved and moved by the start task function.
// - task_state: The state of the task.
// - dependent_tasks: The dependent tasks that should be done before the task
//   starts.
// - following_tasks: The following tasks that are waiting for the task to
//   finish.
// - callback: The callback function. This is the function that will be called
//   when the task is done. Will be retrieved and moved by the start task
//   function.
struct TaskInfo {
  absl::AnyInvocable<absl::StatusOr<Responses>(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)>& callback)>
      task = nullptr;
  TaskState task_state = TaskState::kUnknown;
  absl::flat_hash_set<TaskId> dependent_tasks = {};
  absl::flat_hash_set<TaskId> following_tasks = {};
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;
};

// The execution manager is responsible for managing the execution of the tasks.
// It will handle the scheduling of the tasks and the dependencies between them.
// Note: The execution manager will create its own threadpool for executing the
// tasks, so thread safety interaction should be handled properly.
class ExecutionManager {
 public:
  // Creates an ExecutionManager.
  // The ExecutionManager will take ownership of the executors and the sampler.
  // - tokenizer: The tokenizer used for encoding the text input. This is
  //   expected to be non-null.
  // - llm_executor: The executor used for prefill/decode the LLM. This is
  //   expected to be non-null.
  // - vision_executor: The vision executor used for encoding the image input.
  //   This can be null if no vision modality is used.
  // - audio_executor: The audio executor used for encoding the audio input.
  //   This can be null if no audio modality is used.
  // - sampler: The sampler used for sampling the LLM.
  //   This can be null if no external sampling is needed.
  // - session_config: The session config used for the current execution.
  //   TODO b/409401231 - Move this config into session creation.
  static absl::StatusOr<std::unique_ptr<ExecutionManager>> Create(
      Tokenizer* absl_nonnull tokenizer,
      std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
      std::unique_ptr<VisionExecutor> vision_executor,
      std::unique_ptr<AudioExecutor> audio_executor,
      std::unique_ptr<Sampler> sampler, SessionConfig session_config);

  ~ExecutionManager() = default;

  // Waits until the task is done or the timeout is reached.
  // Returns:
  // - OK if the task is done.
  // - DEADLINE_EXCEEDED if the timeout is reached.
  // - Other errors if the task is failed.
  absl::Status WaitUntilDone(TaskId task_id, absl::Duration timeout);

  // Waits until all tasks are done or the timeout is reached.
  // Returns:
  // - OK if all tasks are done.
  // - DEADLINE_EXCEEDED if the timeout is reached.
  // - Other errors if any of the tasks is failed.
  absl::Status WaitUntilAllDone(absl::Duration timeout);

  // Returns a new task ID.
  // The returned task ID is guaranteed to be unique.
  absl::StatusOr<TaskId> GetNewTaskId();

  // Adds a prefill task to the execution manager.
  // - task_id: The task ID of the task.
  // - inputs: The inputs of the task.
  // - dep_tasks: The dependent tasks that should be done before the task
  //   starts.
  // - benchmark_info: The benchmark info for collecting the performance data.
  // - callback: The callback function.
  // Note: AddPrefillTask will acquire the task lookup mutex.
  absl::Status AddPrefillTask(
      TaskId task_id, std::vector<InputData> inputs,
      absl::flat_hash_set<TaskId> dep_tasks,
      std::optional<BenchmarkInfo>& benchmark_info,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback)
      ABSL_LOCKS_EXCLUDED(task_lookup_mutex_);

  // Adds a decode task to the execution manager.
  // - task_id: The task ID of the task.
  // - dep_tasks: The dependent tasks that should be done before the task
  //   starts.
  // - num_output_candidates: The number of output candidates.
  // - constraint: The constraint for the decode task.
  // - cancelled: The cancelled flag for the decode task.
  // - benchmark_info: The benchmark info for collecting the performance data.
  // - callback: The callback function.
  // Note: AddDecodeTask will acquire the task lookup mutex.
  absl::Status AddDecodeTask(
      TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
      int num_output_candidates, Constraint* absl_nullable constraint,
      std::shared_ptr<std::atomic<bool>> cancelled,
      std::optional<BenchmarkInfo>& benchmark_info,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback)
      ABSL_LOCKS_EXCLUDED(task_lookup_mutex_);

 private:
  // Private constructor. Use the Create function instead.
  ExecutionManager(Tokenizer* absl_nonnull tokenizer,
                   std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
                   std::unique_ptr<VisionExecutor> vision_executor,
                   std::unique_ptr<AudioExecutor> audio_executor,
                   std::unique_ptr<Sampler> sampler,
                   // TODO b/409401231 - Move this config into session creation.
                   const SessionConfig& session_config,
                   std::optional<BenchmarkInfo> benchmark_info)
      : tokenizer_(std::move(tokenizer)),
        llm_executor_(std::move(llm_executor)),
        vision_executor_(std::move(vision_executor)),
        audio_executor_(std::move(audio_executor)),
        sampler_(std::move(sampler)),
        benchmark_info_(std::move(benchmark_info)) {
    stop_token_detector_ = std::make_unique<StopTokenDetector>(1);
    for (const auto& stop_token_sequence : session_config.GetStopTokenIds()) {
      auto status =
          stop_token_detector_->AddStopTokenSequence(stop_token_sequence);
      if (!status.ok()) {
        ABSL_LOG(ERROR) << "Failed to add stop token sequence: " << status;
      }
    }

    execution_thread_pool_ =
        std::make_unique<ThreadPool>(/*name_prefix=*/"execution_thread_pool",
                                     /*max_num_threads=*/1);
  }

  // Creates a task with the given task ID, task, dependent tasks, and callback.
  // - task_id: The task ID of the task.
  // - task: The task function.
  // - dependent_tasks: The dependent tasks that should be done before the task
  //   starts.
  // - callback: The callback function.
  // Note: CreateTask will acquire the task lookup mutex.
  absl::Status CreateTask(
      TaskId task_id,
      absl::AnyInvocable<absl::StatusOr<Responses>(
          absl::AnyInvocable<void(absl::StatusOr<Responses>)>&
              callback)> absl_nonnull task,
      absl::flat_hash_set<TaskId> dependent_tasks,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback)
      ABSL_LOCKS_EXCLUDED(task_lookup_mutex_);

  // Queues the task with the given task ID.
  // - task_id: The task ID of the task.
  // Note: QueueTask expects the callers to acquire the task lookup mutex before
  // calling it.
  absl::Status QueueTask(TaskId task_id)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(task_lookup_mutex_);

  // Update the task states and return required functions for starting the
  // task.
  // Returns:
  // - The first function is the task function.
  // - The second function is the callback function.
  // Note: StartTask will acquire the task lookup mutex.
  absl::StatusOr<std::pair<
      absl::AnyInvocable<absl::StatusOr<Responses>(
          absl::AnyInvocable<void(absl::StatusOr<Responses>)>& callback)>,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
  StartTask(TaskId task_id) ABSL_LOCKS_EXCLUDED(task_lookup_mutex_);

  // Finishes the task with the given task ID, responses, and callback.
  // - task_id: The task ID of the task.
  // - responses: The responses of the task.
  // - callback: The callback function.
  // Note: FinishTask will acquire the task lookup mutex.
  absl::Status FinishTask(
      TaskId task_id, absl::StatusOr<Responses> responses,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback)
      ABSL_LOCKS_EXCLUDED(task_lookup_mutex_);

  // Returns all following tasks that are waiting.
  // - task_id: The task ID of the task.
  // Returns:
  // - The set of following tasks that are waiting for dependent tasks.
  // Note: AllFollowingWaitingTasks expects the callers to acquire the task
  // lookup mutex before calling it.
  absl::StatusOr<absl::flat_hash_set<TaskId>> FollowingWaitingTasks(
      TaskId task_id) ABSL_EXCLUSIVE_LOCKS_REQUIRED(task_lookup_mutex_);

  // Confirms the task state with the given task ID and expected state.
  // - task_id: The task ID of the task.
  // - expected_state: The expected state of the task.
  // - trigger_callback: Whether to trigger the callback function.
  // Note: ConfirmTaskState expects the callers to acquire the task lookup mutex
  // before calling it.
  absl::Status ConfirmTaskState(TaskId task_id, TaskState expected_state,
                                bool trigger_callback = true)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(task_lookup_mutex_);

  // Checks the dependent tasks and updates the pending task.
  // - dep_tasks: The dependent tasks that should be done before the task
  //   starts.
  // - pending_task_id: The pending task ID.
  // Note: CheckDepTasksAndUpdatePendingTask will acquire the task lookup mutex.
  absl::Status CheckDepTasksAndUpdatePendingTask(
      const std::vector<TaskId>& dep_tasks, TaskId pending_task_id)
      ABSL_LOCKS_EXCLUDED(task_lookup_mutex_);

  // Updates the task state with the given task ID and state.
  // - task_id: The task ID of the task.
  // - state: The state of the task.
  // Note: UpdateTaskState expects the callers to acquire the task lookup mutex
  // before calling it.
  absl::Status UpdateTaskState(TaskId task_id, TaskState state)
      ABSL_LOCKS_EXCLUDED(task_lookup_mutex_);

  // Processes and combines the contents of the preprocessed contents.
  // - preprocessed_contents: The preprocessed contents of the task.
  // Returns:
  // - The processed and combined contents of the preprocessed contents.
  absl::StatusOr<ExecutorInputs> ProcessAndCombineContents(
      const std::vector<InputData>& preprocessed_contents);

  // The next unique task ID.
  std::atomic<TaskId> next_task_id_ = -1;

  // The mutex for protecting the task lookup.
  absl::Mutex task_lookup_mutex_;
  // The task lookup map.
  // The key is the task ID.
  // The value is the task info.
  absl::flat_hash_map<TaskId, TaskInfo> task_lookup_
      ABSL_GUARDED_BY(task_lookup_mutex_) = {};

  // TODO b/409401231 - Use LLM Context which is will be wrapped in a session
  // state.
  int last_prefill_token_id_ = 0;

  // The tokenizer used for encoding the text input.
  Tokenizer* absl_nonnull tokenizer_;

  // TODO b/409401231 - Use Resource Manager instead of raw executors.
  // The executor used for prefill/decode the LLM.
  std::unique_ptr<LlmExecutor> absl_nonnull llm_executor_;

  // TODO b/409401231 - Use Resource Manager instead of raw executors.
  // The vision executor used for encoding the image input.
  std::unique_ptr<VisionExecutor> vision_executor_;

  // TODO b/409401231 - Use Resource Manager instead of raw executors.
  // The audio executor used for encoding the audio input.
  std::unique_ptr<AudioExecutor> audio_executor_;

  // The sampler used for sampling the LLM.
  std::unique_ptr<Sampler> sampler_;

  // TODO b/409401231 - Wrap this into Session's state.
  // The stop token detector used for detecting the stop tokens.
  std::unique_ptr<StopTokenDetector> stop_token_detector_;

  // The benchmark info for the current execution.
  std::optional<BenchmarkInfo> benchmark_info_;

  // The thread pool with a single worker thread used for executing the tasks.
  std::unique_ptr<ThreadPool> execution_thread_pool_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_FRAMEWORK_RESOURCE_MANAGEMENT_EXECUTION_MANAGER_H_
