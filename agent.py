797:    token_est = _count_tokens_approx(messages)
798:    threshold = int(config.llm.ctx_window * config.llm.compaction_threshold)
799:    if token_est > threshold:
800:        messages = await compact(messages, config, client, facts_store=facts_store, turn_index=turn_index)