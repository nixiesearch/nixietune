from transformers import PreTrainedModel
from sentence_transformers import SentenceTransformer
from transformers import TrainingArguments
from transformers import Trainer
from typing import List, Dict, Any, Union, Tuple
import torch
from torch import nn
from sentence_transformers import SentenceTransformer


class BiencoderTrainer(Trainer):
    def __init__(
        self,
        *args,
        text_columns: List[str],
        loss: nn.Module,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.text_columns = text_columns
        self.loss = loss
        self.loss.to(self.model.device)

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        # [print(f"{name} = {t.shape}") for name, t in inputs.items()]

        features = self.collect_features(inputs)
        loss = self.loss(features, inputs["label"])
        if return_outputs:
            output = [torch.Tensor()] + features
            return loss, output
        return loss

    def collect_features(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Turn the inputs from the dataloader into the separate model inputs."""
        return [
            {
                "input_ids": inputs[f"{column}_input_ids"],
                "attention_mask": inputs[f"{column}_attention_mask"],
            }
            for column in self.text_columns
        ]


class BiencoderModel(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, model: SentenceTransformer):
        super().__init__(model[0].auto_model.config)
        self.model = model

    def forward(self, tensor):
        return self.model.forward(tensor)
