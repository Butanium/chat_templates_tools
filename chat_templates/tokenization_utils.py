from pathlib import Path
import warnings
import torch as th
import re

from transformers import AutoTokenizer


class IncompleteTokenizerProxy:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        if name == "ctrl_template" and self.tokenizer.ctrl_template is None:
            raise AttributeError(
                "Tokenizer was not patched using tools.tokenization_utils.patch_tokenizer with a control template, so can't be used to compute a control mask"
            )
        elif (
            name
            in [
                "start_of_turn_token",
                "end_of_turn_token",
                "start_of_turn_token_id",
                "end_of_turn_token_id",
            ]
            and getattr(self.tokenizer, name) is None
        ):
            raise AttributeError(
                f"Tokenizer was not patched using tools.tokenization_utils.patch_tokenizer with a {name}."
            )
        return getattr(self.tokenizer, name)

    def __setattr__(self, name, value):
        if name == "tokenizer":
            super().__setattr__(name, value)
            return
        return setattr(self.tokenizer, name, value)


template_path = Path(__file__).parent.parent / "templates"
with open(template_path / "gemma_chat_template.jinja", "r") as f:
    GEMMA_CHAT_TEMPLATE = f.read()
with open(template_path / "llama3.1_chat_template.jinja", "r") as f:
    LLAMA3_1_CHAT_TEMPLATE = f.read()
with open(template_path / "gemma_chat_template_ctrl_tokens.jinja", "r") as f:
    GEMMA_CTRL_TEMPLATE = f.read()
with open(template_path / "llama3.1_chat_template_ctrl_tokens.jinja", "r") as f:
    LLAMA3_1_CTRL_TEMPLATE = f.read()


def patch_tokenizer(
    tokenizer,
    model_name: str,
    ctrl_template: str | None = None,
    chat_template: str | None = None,
    end_of_turn_token: str | None = None,
    start_of_turn_token: str | None = None,
    pad_token: str | None = None,
):
    if "gemma-2" in model_name.lower():
        if chat_template is None:
            chat_template = GEMMA_CHAT_TEMPLATE
        if ctrl_template is None:
            ctrl_template = GEMMA_CTRL_TEMPLATE
        if start_of_turn_token is None:
            start_of_turn_token = "<start_of_turn>"
        if end_of_turn_token is None:
            end_of_turn_token = "<end_of_turn>"
    elif (
        "meta-llama/Meta-Llama-3.1".lower() in model_name.lower()
        or "meta-llama/Llama-3.2".lower() in model_name.lower()
    ):
        if chat_template is None:
            chat_template = LLAMA3_1_CHAT_TEMPLATE
        if ctrl_template is None:
            ctrl_template = LLAMA3_1_CTRL_TEMPLATE
        if end_of_turn_token is None:
            end_of_turn_token = "<|eot_id|>"
        if start_of_turn_token is None:
            start_of_turn_token = "<|start_header_id|>"
        if pad_token is None:
            pad_token = "<|end_of_text|>"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = pad_token or tokenizer.eos_token or tokenizer.bos_token
        if pad_token is not None:
            pad_token_id = tokenizer.encode(pad_token, add_special_tokens=False)
            if len(pad_token_id) != 1:
                raise ValueError(f"Pad token must be a single token: {pad_token_id}")
            tokenizer.pad_token_id = pad_token_id[0]
        else:
            tokenizer.pad_token_id = (
                tokenizer.eos_token_id
                if tokenizer.eos_token is not None
                else tokenizer.bos_token_id
            )
        if tokenizer.pad_token_id is None:
            raise ValueError("Pad token couldn't be set automatically")
    use_proxy = False
    if chat_template is not None:
        tokenizer.chat_template = chat_template
    if ctrl_template is None:
        warnings.warn(
            f"No control template provided, you won't be able to use the control token mask for {model_name}"
        )
        tokenizer.ctrl_template = None
        use_proxy = True
    else:
        tokenizer.ctrl_template = ctrl_template
    if tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer has no chat template, please provide one in the tokenizer_kwargs"
        )
    generation_pattern = re.compile(r"\{%\s*generation\s*%\}")
    if not generation_pattern.search(tokenizer.chat_template):
        raise ValueError(
            f"Chat template for {model_name}"
            " does not contain {% generation %} keyword"
        )
    tokenizer.end_of_turn_token = end_of_turn_token
    if end_of_turn_token is None:
        warnings.warn(
            "No end of turn token provided, you won't be able to use tokenizer.end_of_turn_token"
        )
        use_proxy = True
        tokenizer.end_of_turn_token_id = None
    else:
        id = tokenizer.encode(end_of_turn_token, add_special_tokens=False)
        if len(id) != 1:
            raise ValueError(f"end of turn token must be a single token: {id}")
        tokenizer.end_of_turn_token_id = id[0]
    tokenizer.start_of_turn_token = start_of_turn_token
    if start_of_turn_token is None:
        warnings.warn(
            "No start of turn token provided, you won't be able to use tokenizer.start_of_turn_token"
        )
        use_proxy = True
        tokenizer.start_of_turn_token_id = None
    else:
        id = tokenizer.encode(start_of_turn_token, add_special_tokens=False)
        if len(id) != 1:
            raise ValueError(f"start of turn token must be a single token: {id}")
        tokenizer.start_of_turn_token_id = id[0]

    if use_proxy:
        tokenizer = IncompleteTokenizerProxy(tokenizer)
    return tokenizer


def tokenize_with_ctrl_mask(
    convs: list[list[dict[str, str]]],
    tokenizer,
    **tokenizer_kwargs,
) -> dict:
    """
    Tokenizes conversations with a control mask indicating chat control tokens.

    This function tokenizes a list of conversations using a custom template for chat control tokens. It returns a dictionary containing the tokenized conversations, attention mask, control mask, and assistant masks.

    Args:
        convs (list[list[dict[str, str]]]): A list of conversations, where each conversation is a list of dictionaries containing 'role' and 'content'.
        tokenizer: The tokenizer to use for tokenization.
        **tokenizer_kwargs: Additional keyword arguments to pass to the tokenizer.

    Returns:
        dict: A dictionary containing the tokenized conversations, attention_mask, ctrl_mask, and assistant_masks
    """
    # Update tokenizer_kwargs with default settings for control mask and attention mask
    kwargs = tokenizer_kwargs.copy()
    kwargs.update(
        dict(
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            return_dict=True,
            chat_template=tokenizer.ctrl_template,
        )
    )
    ctrl_tok_dict = tokenizer.apply_chat_template(
        convs,
        **kwargs,
    )
    ctrl_mask = th.tensor(ctrl_tok_dict["assistant_masks"], dtype=th.bool)
    tokenizer_kwargs["return_dict"] = True
    tokenizer_kwargs["return_assistant_tokens_mask"] = True
    tokenizer_kwargs["return_tensors"] = "pt"
    if tokenizer.chat_template is None and "chat_template" not in tokenizer_kwargs:
        raise ValueError(
            "Tokenizer has no chat template, please provide one in the tokenizer_kwargs"
        )
    else:
        chat_template = tokenizer_kwargs.get("chat_template", tokenizer.chat_template)
        if "generation" not in chat_template:
            raise ValueError("Chat template does not contain {% generation %} keyword")
    tok_dict = tokenizer.apply_chat_template(convs, **tokenizer_kwargs)
    if tok_dict["attention_mask"].shape != ctrl_tok_dict["attention_mask"].shape:
        raise ValueError(
            f"attention_mask shapes are not the same: {tok_dict['attention_mask'].shape} != {ctrl_tok_dict['attention_mask'].shape}\n"
            "This means your chat template is not the same as the control template"
        )
    tok_dict["ctrl_mask"] = ctrl_mask
    tok_dict["assistant_masks"] = th.tensor(tok_dict["assistant_masks"], dtype=th.bool)
    return tok_dict


def tokenize_with_ctrl_ids(
    convs: list[list[dict[str, str]]],
    tokenizer,
    **tokenizer_kwargs,
) -> dict:
    """
    Same as tokenize_with_ctrl_mask, but labels the control tokens from 1 to 10 instead all True
    """
    tok_dict = tokenize_with_ctrl_mask(convs, tokenizer, **tokenizer_kwargs)
    mask = tok_dict["ctrl_mask"]
    ids = mask.to(th.int)
    n_ctrl_toks = ids.sum()
    rep_1_10 = th.arange(1, 11, dtype=th.int).repeat(n_ctrl_toks // 10 + 1)[
        :n_ctrl_toks
    ]
    ids[mask] = rep_1_10
    tok_dict["ctrl_ids"] = ids
    return tok_dict
