#include "runtime/components/lora.h"

#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/litert_status_util.h"
#include "runtime/util/lora_data.h"
#include "runtime/util/lora_util.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

namespace {

// Names of the signature runners, used to get the signature runners from the
// interpreter.
// TODO: b/450616365 - Consolidate constant definitions.
constexpr char kDecodeSignatureRunner[] = "decode";

}  // namespace

absl::StatusOr<std::unique_ptr<LoRA>> LoRA::Create(
    std::unique_ptr<LoraData> lora_data, const litert::Model& model,
    const litert::CompiledModel& compiled_model) {
  auto lora =
      absl::WrapUnique(new LoRA(std::move(lora_data), model, compiled_model));
  RETURN_IF_ERROR(lora->Init());
  return lora;
}

absl::Status LoRA::Init() {
  // Get the input names from the default signature.
  LITERT_ASSIGN_OR_RETURN_ABSL(
      auto input_names, model_.GetSignatureInputNames(kDecodeSignatureRunner));

  for (const auto& input_name : input_names) {
    if (!IsLoRAInputName(input_name)) {
      continue;
    }
    // Create the input buffer for the LoRA tensor.
    LITERT_ASSIGN_OR_RETURN_ABSL(
        litert::TensorBuffer tensor_buffer,
        compiled_model_.CreateInputBuffer(kDecodeSignatureRunner, input_name));

    LITERT_ASSIGN_OR_RETURN_ABSL(
        auto lock_and_addr, litert::TensorBufferScopedLock::Create(
                                tensor_buffer, TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN_ABSL(auto tensor_buffer_size,
                                 tensor_buffer.PackedSize());

    if (lora_data_->HasTensor(input_name)) {
      // Read the tensor data from LoraData.
      ASSIGN_OR_RETURN(auto lora_tensor_data,
                       lora_data_->ReadTensor(input_name));

      // Copy the data from LoraData to the TensorBuffer.
      RET_CHECK_EQ(tensor_buffer_size, lora_tensor_data->Size())
          << "LoRA tensor size mismatch between model input and Lora Data: "
          << tensor_buffer_size << " vs. " << lora_tensor_data->Size();
      std::memcpy(lock_and_addr.second, lora_tensor_data->Data(),
                  lora_tensor_data->Size());
    } else {
      // Fill the buffer with zeros if the tensor is not in LoraData.
      std::memset(lock_and_addr.second, 0, tensor_buffer_size);
    }

    lora_buffers_[input_name] = std::move(tensor_buffer);
  }
  return absl::OkStatus();
}

absl::StatusOr<litert::TensorBuffer> LoRA::GetLoRABuffer(
    const std::string& name) const {
  auto it = lora_buffers_.find(name);
  if (it == lora_buffers_.end()) {
    return absl::NotFoundError("LoRA tensor not found.");
  }
  LITERT_ASSIGN_OR_RETURN_ABSL(auto duplicated_buffer, it->second.Duplicate());
  return duplicated_buffer;
}

absl::StatusOr<absl::flat_hash_map<absl::string_view, litert::TensorBuffer>>
LoRA::GetLoRABuffers() const {
  absl::flat_hash_map<absl::string_view, litert::TensorBuffer> buffers;
  for (const auto& [name, buffer] : lora_buffers_) {
    LITERT_ASSIGN_OR_RETURN_ABSL(buffers[name], buffer.Duplicate());
  }
  return buffers;
}

}  // namespace litert::lm
