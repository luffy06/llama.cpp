#include "common.h"

#include "console.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static gpt_params               * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;

static void llama_log_callback_logTee(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    LOG_TEE("%s", text);
}

int main(int argc, char ** argv) {
    gpt_params params;
    g_params = &params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }
    llama_sampling_params & sparams = params.sparams;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("main", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
    llama_log_set(llama_log_callback_logTee, nullptr);
#endif // LOG_DISABLE_LOGS

    // TODO: Dump params ?
    //LOG("Params perplexity: %s\n", LOG_TOSTR(params.perplexity));

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.logits_all) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_TEE("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_TEE("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_TEE("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }
    
    LOG_TEE("%s: build = %d (%s)\n",      __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG_TEE("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    std::mt19937 rng(params.seed);
    if (params.random_prompt || params.prompt_list.empty()) {
        LOG("%s: randomly generate 100 prompts\n", __func__);
        for (uint32_t i = 0; i < 100; ++ i) {
            params.prompt_list.push_back(gpt_random_prompt(rng));
        }
    }

    LOG("%s: llama backend init\n", __func__);
    llama_backend_init(params.numa);

    llama_model * model;
    llama_context * ctx;
    g_model = &model;
    g_ctx = &ctx;

    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);
    LOG("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train) {
        LOG_TEE("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        LOG_TEE("\n");
        LOG_TEE("%s\n", get_system_info(params).c_str());
    }

    const bool add_bos = llama_should_add_bos_token(model);
    LOG("add_bos: %d\n", add_bos);

    std::vector<std::vector<llama_token>> embd_inp_list;

    assert(!params.prompt_list.empty());
    LOG_TEE("%s: tokenize %d prompt\n", __func__, params.prompt_list.size());
    for (uint32_t i = 0; i < params.prompt_list.size(); ++ i) {
        embd_inp_list.push_back(::llama_tokenize(ctx, params.prompt_list[i], add_bos, true));
    }

    std::vector<llama_timings> all_timings;
    uint32_t step = embd_inp_list.size() / 10;
    // Should not run without any tokens
    for (uint32_t i = 0; i < embd_inp_list.size(); ++ i) {
        std::vector<llama_token> & embd_inp = embd_inp_list[i];
        if (i % step == 0 || i == embd_inp_list.size() - 1) {
            LOG_TEE("%s: Process (%d/%d) prompt\n", __func__, i, embd_inp_list.size());
        }
        if (embd_inp.empty()) {
            embd_inp.push_back(llama_token_bos(model));
            LOG("%dth embd_inp was considered empty and bos was added: %s\n", i, LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
        }

        if ((int) embd_inp.size() > n_ctx - 4) {
            LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
            return 1;
        }


        LOGLN("recalculate the cached logits (check): embd_inp.empty() %s, embd_inp.size() %zu", log_tostr(embd_inp.empty()), embd_inp.size());
        // number of tokens to keep when resetting context
        if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct || params.chatml) {
            params.n_keep = (int)embd_inp.size();
        }

        if (params.verbose_prompt) {
            LOG_TEE("\n");
            LOG_TEE("%s: prompt: '%s'\n", __func__, params.prompt_list[i].c_str());
            LOG_TEE("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
            for (int i = 0; i < (int) embd_inp.size(); i++) {
                LOG_TEE("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
            }

            if (params.n_keep > 0) {
            LOG_TEE("%s: static prompt based on n_keep: '", __func__);
                for (int i = 0; i < params.n_keep; i++) {
                    LOG_TEE("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
                }
                LOG_TEE("'\n");
            }
            LOG_TEE("\n");
        }

        // LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
        // LOG_TEE("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
        // LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
        // LOG_TEE("\n\n");

        // bool input_echo           = true;

        int n_past             = 0;
        int n_remain           = params.n_predict;
        int n_consumed         = 0;

        std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
        std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
        std::ostringstream output_ss;     g_output_ss     = &output_ss;

        // the first thing we will do is to output the prompt, so set color accordingly
        console::set_display(console::prompt);

        std::vector<llama_token> embd;
        
        struct llama_sampling_context * ctx_sampling = llama_sampling_init(sparams);
    
        llama_reset_timings(ctx);

        while (n_remain != 0) {
            // predict
            if (!embd.empty()) {
                // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
                // --prompt or --file which uses the same value.
                int max_embd_size = n_ctx - 4;

                // Ensure the input doesn't exceed the context size by truncating embd if necessary.
                if ((int) embd.size() > max_embd_size) {
                    const int skipped_tokens = (int) embd.size() - max_embd_size;
                    embd.resize(max_embd_size);

                    console::set_display(console::error);
                    printf("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                    console::set_display(console::reset);
                    fflush(stdout);
                }

                // infinite text generation via context swapping
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
                if (n_past + (int) embd.size() > n_ctx) {
                    if (params.n_predict == -2) {
                        LOG_TEE("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left    = n_past - params.n_keep - 1;
                    const int n_discard = n_left/2;

                    LOG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                        n_past, n_left, n_ctx, params.n_keep, n_discard);

                    llama_kv_cache_seq_rm   (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);
                    llama_kv_cache_seq_shift(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    LOG("after swap: n_past = %d\n", n_past);

                    LOG("embd: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());
                }

                // evaluate tokens in batches
                // embd is typically prepared beforehand to fit within a batch, but not always
                for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                    int n_eval = (int) embd.size() - i;
                    if (n_eval > params.n_batch) {
                        n_eval = params.n_batch;
                    }

                    LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

                    if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                        LOG_TEE("%s : failed to eval\n", __func__);
                        return 1;
                    }

                    n_past += n_eval;

                    LOG("n_past = %d\n", n_past);
                }

            }

            embd.clear();

            if ((int) embd_inp.size() <= n_consumed) {
                const llama_token id = llama_sampling_sample(ctx_sampling, ctx, NULL);

                llama_sampling_accept(ctx_sampling, ctx, id, true);

                LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, ctx_sampling->prev).c_str());

                embd.push_back(id);

                // // echo this to console
                // input_echo = true;

                // decrement remaining sampling budget
                --n_remain;

                LOG("n_remain: %d\n", n_remain);
            } else {
                // some user input remains from prompt or interaction, forward it to processing
                LOG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
                while ((int) embd_inp.size() > n_consumed) {
                    embd.push_back(embd_inp[n_consumed]);

                    // push the prompt in the sampling context in order to apply repetition penalties later
                    // for the prompt, we don't apply grammar rules
                    llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);

                    ++n_consumed;
                    if ((int) embd.size() >= params.n_batch) {
                        break;
                    }
                }
            }

            // // display text
            // if (input_echo) {
            //     for (auto id : embd) {
            //         const std::string token_str = llama_token_to_piece(ctx, id);
            //         printf("%s", token_str.c_str());

            //         if (embd.size() > 1) {
            //             input_tokens.push_back(id);
            //         } else {
            //             output_tokens.push_back(id);
            //             output_ss << token_str;
            //         }
            //     }
            //     fflush(stdout);
            // }
            // // reset color to default if there is no pending user input
            // if (input_echo && (int) embd_inp.size() == n_consumed) {
            //     console::set_display(console::reset);
            // }

            // if not currently processing queued inputs;
            if ((int) embd_inp.size() <= n_consumed) {
                // deal with end of text token in interactive mode
                if (llama_sampling_last(ctx_sampling) == llama_token_eos(model)) {
                    LOG("found EOS token\n");
                }
            }

            // end of text token
            if (!embd.empty() && embd.back() == llama_token_eos(model)) {
                LOG_TEE(" [end of text]\n");
                break;
            }
        }

        all_timings.push_back(llama_get_timings(ctx));
        llama_sampling_free(ctx_sampling);
    }

    llama_timings avg_timings;
    avg_timings.t_load_ms = 0;
    avg_timings.t_sample_ms = avg_timings.n_sample = 0;
    avg_timings.t_p_eval_ms = avg_timings.n_p_eval = 0;
    avg_timings.t_eval_ms = avg_timings.n_eval = 0;
    for (uint32_t i = 0; i < all_timings.size(); ++ i) {
        avg_timings.t_load_ms += all_timings[i].t_load_ms;
        avg_timings.t_sample_ms += all_timings[i].t_sample_ms;
        avg_timings.n_sample += all_timings[i].n_sample;
        avg_timings.t_p_eval_ms += all_timings[i].t_p_eval_ms;
        avg_timings.n_p_eval += all_timings[i].n_p_eval;
        avg_timings.t_eval_ms += all_timings[i].t_eval_ms;
        avg_timings.n_eval += all_timings[i].n_eval;
    }
    LOG_TEE("\n");
    LOG_TEE("%s:        load time = %10.2f ms\n", __func__, avg_timings.t_load_ms / all_timings.size());
    LOG_TEE("%s:      sample time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, avg_timings.t_sample_ms, avg_timings.n_sample, avg_timings.t_sample_ms / avg_timings.n_sample, 1e3 / avg_timings.t_sample_ms * avg_timings.n_sample);
    LOG_TEE("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, avg_timings.t_p_eval_ms, avg_timings.n_p_eval, avg_timings.t_p_eval_ms / avg_timings.n_p_eval, 1e3 / avg_timings.t_p_eval_ms * avg_timings.n_p_eval);
    LOG_TEE("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, avg_timings.t_eval_ms, avg_timings.n_eval, avg_timings.t_eval_ms / avg_timings.n_eval, 1e3 / avg_timings.t_eval_ms * avg_timings.n_eval);
    LOG_TEE("%s:       total time = %10.2f ms\n", __func__, (avg_timings.t_end_ms - avg_timings.t_start_ms));

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

#ifndef LOG_DISABLE_LOGS
    LOG_TEE("Log end\n");
#endif // LOG_DISABLE_LOGS
    return 0;
}
