#include "api_export.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>

WavFileWrapper WavFileWrapper::load_wav_file(const char *filename) {
  std::vector<float> pcmf32;
  std::vector<std::vector<float>> pcmf32s;
  if (!::read_wav(filename, pcmf32, pcmf32s, false)) {
    throw std::runtime_error("Failed to load wav file");
  }
  return WavFileWrapper(&pcmf32, &pcmf32s);
}

namespace py = pybind11;
using namespace pybind11::literals;

typedef std::function<void(Context &, int, py::object &)> NewSegmentCallback;

namespace whisper {

PYBIND11_MODULE(api, m) {
  m.doc() = "Python interface for whisper.cpp";

  // NOTE: default attributes
  m.attr("SAMPLE_RATE") = py::int_(WHISPER_SAMPLE_RATE);
  m.attr("N_FFT") = py::int_(WHISPER_N_FFT);
  m.attr("N_MEL") = py::int_(WHISPER_N_MEL);
  m.attr("HOP_LENGTH") = py::int_(WHISPER_HOP_LENGTH);
  m.attr("CHUNK_SIZE") = py::int_(WHISPER_CHUNK_SIZE);

  // NOTE: export Context API
  ExportContextApi(m);

  m.def("load_wav_file", &WavFileWrapper::load_wav_file, "filename"_a,
        py::return_value_policy::reference);

  py::class_<WavFileWrapper>(m, "Wavfile",
                             "A light wrapper for the processed wav file.")
      .def_property_readonly(
          "stereo", [](WavFileWrapper &self) { return self.stereo; },
          py::return_value_policy::reference)
      .def_property_readonly(
          "mono", [](WavFileWrapper &self) { return self.mono; },
          py::return_value_policy::reference);

  // NOTE: export Params API
  py::enum_<SamplingStrategies::StrategyType>(m, "StrategyType")
      .value("SAMPLING_GREEDY", SamplingStrategies::GREEDY)
      .value("SAMPLING_BEAM_SEARCH", SamplingStrategies::BEAM_SEARCH)
      .export_values();

  py::class_<SamplingGreedy>(m, "SamplingGreedyStrategy")
      .def(py::init<>())
      .def_readwrite("best_of", &SamplingGreedy::best_of);

  py::class_<SamplingBeamSearch>(m, "SamplingBeamSearchStrategy")
      .def(py::init<>())
      .def_readwrite("beam_size", &SamplingBeamSearch::beam_size)
      .def_readwrite("patience", &SamplingBeamSearch::patience);

  py::class_<SamplingStrategies>(m, "SamplingStrategies",
                                 "Available sampling strategy for whisper")
      .def_static("from_strategy_type", &SamplingStrategies::from_strategy_type,
                  "strategy"_a)
      .def_readwrite("type", &SamplingStrategies::type)
      .def_readwrite("greedy", &SamplingStrategies::greedy)
      .def_readwrite("beam_search", &SamplingStrategies::beam_search);

  py::class_<Params>(m, "Params", "Whisper parameters container")
      .def_static("from_sampling_strategy", &Params::from_sampling_strategy,
                  "sampling_strategy"_a)
      .def_property("num_threads", &Params::get_n_threads,
                    &Params::set_n_threads)
      .def_property("num_max_text_ctx", &Params::get_n_max_text_ctx,
                    &Params::set_n_max_text_ctx)
      .def_property("offset_ms", &Params::get_offset_ms, &Params::set_offset_ms)
      .def_property("duration_ms", &Params::get_duration_ms,
                    &Params::set_duration_ms)
      .def_property("translate", &Params::get_translate, &Params::set_translate)
      .def_property("no_context", &Params::get_no_context,
                    &Params::set_no_context)
      .def_property("single_segment", &Params::get_single_segment,
                    &Params::set_single_segment)
      .def_property("print_special", &Params::get_print_special,
                    &Params::set_print_special)
      .def_property("print_progress", &Params::get_print_progress,
                    &Params::set_print_progress)
      .def_property("print_realtime", &Params::get_print_realtime,
                    &Params::set_print_realtime)
      .def_property("print_timestamps", &Params::get_print_timestamps,
                    &Params::set_print_timestamps)
      .def_property("token_timestamps", &Params::get_token_timestamps,
                    &Params::set_token_timestamps)
      .def_property("timestamp_token_probability_threshold",
                    &Params::get_thold_pt, &Params::set_thold_pt)
      .def_property("timestamp_token_sum_probability_threshold",
                    &Params::get_thold_ptsum, &Params::set_thold_ptsum)
      .def_property("max_segment_length", &Params::get_max_len,
                    &Params::set_max_len)
      .def_property("split_on_word", &Params::get_split_on_word,
                    &Params::set_split_on_word)
      .def_property("max_tokens", &Params::get_max_tokens,
                    &Params::set_max_tokens)
      .def_property("speed_up", &Params::get_speed_up, &Params::set_speed_up)
      .def_property("audio_ctx", &Params::get_audio_ctx, &Params::set_audio_ctx)
      .def("set_tokens", &Params::set_tokens, "tokens"_a)
      .def_property_readonly("prompt_tokens", &Params::get_prompt_tokens)
      .def_property_readonly("prompt_num_tokens", &Params::get_prompt_n_tokens)
      .def_property("language",
                    py::cpp_function(
                        [](Params &self) {
                          const char *language = self.get_language();
                          return std::string(language);
                        },
                        py::return_value_policy::copy),
                    py::cpp_function(
                        [](Params &self, const char *language) {
                          self.set_language(language);
                        },
                        py::keep_alive<1, 2>()))
      .def_property("suppress_blank", &Params::get_suppress_blank,
                    &Params::set_suppress_blank)
      .def_property("suppress_none_speech_tokens",
                    &Params::get_suppress_none_speech_tokens,
                    &Params::set_suppress_none_speech_tokens)
      .def_property("temperature", &Params::get_temperature,
                    &Params::set_temperature)
      .def_property("max_intial_timestamps", &Params::get_max_intial_ts,
                    &Params::set_max_intial_ts)
      .def_property("length_penalty", &Params::get_length_penalty,
                    &Params::set_length_penalty)
      .def_property("temperature_inc", &Params::get_temperature_inc,
                    &Params::set_temperature_inc)
      .def_property("entropy_threshold", &Params::get_entropy_thold,
                    &Params::set_entropy_thold)
      .def_property("logprob_threshold", &Params::get_logprob_thold,
                    &Params::set_logprob_thold)
      .def_property("no_speech_threshold", &Params::get_no_speech_thold,
                    &Params::set_no_speech_thold)
      .def("on_new_segment",
           [](Params &self, NewSegmentCallback & callback, py::object & user_data) {
             using namespace std::placeholders;
             self.set_new_segment_callback(
               std::bind([](NewSegmentCallback & callback, py::object & user_data,
                            Context & ctx, int n_new) mutable {
                   (callback)(ctx, n_new, user_data);
             }, std::move(callback), std::move(user_data), _1, _2));
           },
           "callback"_a, "user_data"_a = py::none(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
    .def("__repr__", [](Params &self) {
      std::stringstream s;
        s << "Params(" << std::endl
<<       "strategy = "                       <<  self.wfp.strategy << std::endl
<<       "n_threads = "                      <<  self.wfp.n_threads << std::endl
<<       "n_max_text_ctx = "                 <<  self.wfp.n_max_text_ctx << std::endl
<<       "offset_ms = "                      <<  self.wfp.offset_ms << std::endl
<<       "duration_ms = "                    <<  self.wfp.duration_ms << std::endl
<<       "translate = "                      <<  self.wfp.translate << std::endl
<<       "no_context = "                     <<  self.wfp.no_context << std::endl
<<       "single_segment = "                 <<  self.wfp.single_segment << std::endl
<<       "print_special = "                  <<  self.wfp.print_special << std::endl
<<       "print_progress = "                 <<  self.wfp.print_progress << std::endl
<<       "print_realtime = "                 <<  self.wfp.print_realtime << std::endl
<<       "print_timestamps = "               <<  self.wfp.print_timestamps << std::endl
<<       "token_timestamps = "               <<  self.wfp.token_timestamps << std::endl
<<       "thold_pt = "                       <<  self.wfp.thold_pt << std::endl
<<       "thold_ptsum = "                    <<  self.wfp.thold_ptsum << std::endl
<<       "max_len = "                        <<  self.wfp.max_len << std::endl
<<       "split_on_word = "                  <<  self.wfp.split_on_word << std::endl
<<       "max_tokens = "                     <<  self.wfp.max_tokens << std::endl
<<       "speed_up = "                       <<  self.wfp.speed_up << std::endl
<<       "audio_ctx = "                      <<  self.wfp.audio_ctx << std::endl
<<       "prompt_n_tokens = "                <<  self.wfp.prompt_n_tokens << std::endl
<<       "language = "                       <<  self.wfp.language << std::endl
<<       "suppress_blank = "                 <<  self.wfp.suppress_blank << std::endl
<<       "suppress_non_speech_tokens = "     <<  self.wfp.suppress_non_speech_tokens << std::endl
<<       "temperature = "                    <<  self.wfp.temperature << std::endl
<<       "max_initial_ts = "                 <<  self.wfp.max_initial_ts << std::endl
<<       "length_penalty = "                 <<  self.wfp.length_penalty << std::endl
<<       "temperature_inc = "                <<  self.wfp.temperature_inc << std::endl
<<       "entropy_thold = "                  <<  self.wfp.entropy_thold << std::endl
<<       "logprob_thold = "                  <<  self.wfp.logprob_thold << std::endl
<<       "no_speech_thold = "                <<  self.wfp.no_speech_thold << std::endl
<<       "greedy.best_of = "                 <<  self.wfp.greedy.best_of << std::endl
<<       "beam_search.beam_size = "          <<  self.wfp.beam_search.beam_size << std::endl
<<       "beam_search.patience = "           <<  self.wfp.beam_search.patience << std::endl
          << ")";
        return s.str();
    });
  // TODO: encoder_begin_callback and logits_filter_callback are still missing
}
}; // namespace whisper
