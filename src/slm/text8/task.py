from slm.slm.core import predict_log_probs, sample_forward, sample_steps, training_loss


class Text8Task:
    def __init__(self, config):
        self.config = config

    def prepare(self, clean_tokens):
        steps = sample_steps(len(clean_tokens), self.config.T, device=clean_tokens.device)
        candidates = sample_forward(clean_tokens, steps.float() / self.config.T, self.config.vocab)
        return clean_tokens, candidates, steps

    def batch_size(self, prepared):
        return len(prepared[0])

    def slice_batch(self, prepared, start, stop):
        return tuple(item[start:stop] for item in prepared)

    def loss(self, model, microbatch):
        clean_tokens, candidates, steps = microbatch
        log_probs = predict_log_probs(model, candidates, steps.float() / self.config.T)
        return training_loss(log_probs, clean_tokens, steps, self.config.T, self.config.objective).mean()
