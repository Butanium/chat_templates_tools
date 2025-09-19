## Features
todo, but `tokenization_utils.py` was initially created for gemma 2 only, so there are some functions with very generic docs but that will actually only work for gemma 2. I need to fix this :)

## Templates
The `templates` folder contains the following templates:
- `{model_name}_chat_template.jinja`: Chat template for `model_name` with assistant mask implemented.
- `{model_name}_chat_template_ctrl_tokens.jinja`: Chat template for `model_name` where assistant mask returns 1 for the template tokens.
- `{model_name}_force_thinking.jinja`: Chat template for `model_name`, where `enable_thinking=True` in `tokenizer.apply_chat_template` force the model to think.