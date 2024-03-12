from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class EvaluateFirstStepCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_evaluate = True
