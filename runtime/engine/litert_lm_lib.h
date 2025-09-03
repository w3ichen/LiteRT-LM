#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_LIB_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_LIB_H_

#include <string>

#include "absl/status/status.h"  // from @com_google_absl

namespace litert {
namespace lm {

struct LiteRtLmSettings {
  std::string backend = "gpu";
  std::string sampler_backend = "";
  std::string model_path;
  std::string input_prompt = "What is the tallest building in the world?";
  bool benchmark = false;
  int benchmark_prefill_tokens = 0;
  int benchmark_decode_tokens = 0;
  bool async = true;
  bool report_peak_memory_footprint = false;
  bool force_f32 = false;
  bool multi_turns = false;
  int num_cpu_threads = 0;
};

absl::Status RunLiteRtLm(const LiteRtLmSettings& settings);

}  // namespace lm
}  // namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_LITERT_LM_LIB_H_
