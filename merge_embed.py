import torch
import os
import folder_paths
from safetensors.torch import save_file, load_file
import re
from typing import Union, List, Tuple, Literal, TypedDict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StrEnum(str, Enum):
    """Base class for string enums. Python's StrEnum is not available until 3.11."""

    def __str__(self) -> str:
        return self.value

class IO(StrEnum):
    """Node input/output data types.

    Includes functionality for ``"*"`` (`ANY`) and ``"MULTI,TYPES"``.
    """

    STRING = "STRING"
    IMAGE = "IMAGE"
    MASK = "MASK"
    LATENT = "LATENT"
    BOOLEAN = "BOOLEAN"
    INT = "INT"
    FLOAT = "FLOAT"
    CONDITIONING = "CONDITIONING"
    SAMPLER = "SAMPLER"
    SIGMAS = "SIGMAS"
    GUIDER = "GUIDER"
    NOISE = "NOISE"
    CLIP = "CLIP"
    CONTROL_NET = "CONTROL_NET"
    VAE = "VAE"
    MODEL = "MODEL"
    CLIP_VISION = "CLIP_VISION"
    CLIP_VISION_OUTPUT = "CLIP_VISION_OUTPUT"
    STYLE_MODEL = "STYLE_MODEL"
    GLIGEN = "GLIGEN"
    UPSCALE_MODEL = "UPSCALE_MODEL"
    AUDIO = "AUDIO"
    WEBCAM = "WEBCAM"
    POINT = "POINT"
    FACE_ANALYSIS = "FACE_ANALYSIS"
    BBOX = "BBOX"
    SEGS = "SEGS"

    ANY = "*"
    """Always matches any type, but at a price.

    Causes some functionality issues (e.g. reroutes, link types), and should be avoided whenever possible.
    """
    NUMBER = "FLOAT,INT"
    """A float or an int - could be either"""
    PRIMITIVE = "STRING,FLOAT,INT,BOOLEAN"
    """Could be any of: string, float, int, or bool"""

    def __ne__(self, value: object) -> bool:
        if self == "*" or value == "*":
            return False
        if not isinstance(value, str):
            return True
        a = frozenset(self.split(","))
        b = frozenset(value.split(","))
        return not (b.issubset(a) or a.issubset(b))

class InputTypeOptions(TypedDict):
    """Provides type hinting for the return type of the INPUT_TYPES node function.

    Due to IDE limitations with unions, for now all options are available for all types (e.g. `label_on` is hinted even when the type is not `IO.BOOLEAN`).

    Comfy Docs: https://docs.comfy.org/essentials/custom_node_datatypes
    """

    default: bool | str | float | int | list | tuple
    """The default value of the widget"""
    defaultInput: bool
    """Defaults to an input slot rather than a widget"""
    forceInput: bool
    """`defaultInput` and also don't allow converting to a widget"""
    lazy: bool
    """Declares that this input uses lazy evaluation"""
    rawLink: bool
    """When a link exists, rather than receiving the evaluated value, you will receive the link (i.e. `["nodeId", <outputIndex>]`). Designed for node expansion."""
    tooltip: str
    """Tooltip for the input (or widget), shown on pointer hover"""
    # class InputTypeNumber(InputTypeOptions):
    # default: float | int
    min: float
    """The minimum value of a number (``FLOAT`` | ``INT``)"""
    max: float
    """The maximum value of a number (``FLOAT`` | ``INT``)"""
    step: float
    """The amount to increment or decrement a widget by when stepping up/down (``FLOAT`` | ``INT``)"""
    round: float
    """Floats are rounded by this value (``FLOAT``)"""
    # class InputTypeBoolean(InputTypeOptions):
    # default: bool
    label_on: str
    """The label to use in the UI when the bool is True (``BOOLEAN``)"""
    label_on: str
    """The label to use in the UI when the bool is False (``BOOLEAN``)"""
    # class InputTypeString(InputTypeOptions):
    # default: str
    multiline: bool
    """Use a multiline text box (``STRING``)"""
    placeholder: str
    """Placeholder text to display in the UI when empty (``STRING``)"""
    # Deprecated:
    # defaultVal: str
    dynamicPrompts: bool
    """Causes the front-end to evaluate dynamic prompts (``STRING``)"""

class HiddenInputTypeDict(TypedDict):
    """Provides type hinting for the hidden entry of node INPUT_TYPES."""

    node_id: Literal["UNIQUE_ID"]
    """UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side. It is commonly used in client-server communications (see messages)."""
    unique_id: Literal["UNIQUE_ID"]
    """UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side. It is commonly used in client-server communications (see messages)."""
    prompt: Literal["PROMPT"]
    """PROMPT is the complete prompt sent by the client to the server. See the prompt object for a full description."""
    extra_pnginfo: Literal["EXTRA_PNGINFO"]
    """EXTRA_PNGINFO is a dictionary that will be copied into the metadata of any .png files saved. Custom nodes can store additional information in this dictionary for saving (or as a way to communicate with a downstream node)."""
    dynprompt: Literal["DYNPROMPT"]
    """DYNPROMPT is an instance of comfy_execution.graph.DynamicPrompt. It differs from PROMPT in that it may mutate during the course of execution in response to Node Expansion."""

class InputTypeDict(TypedDict):
    """Provides type hinting for node INPUT_TYPES.

    Comfy Docs: https://docs.comfy.org/essentials/custom_node_more_on_inputs
    """

    required: dict[str, tuple[IO, InputTypeOptions]]
    """Describes all inputs that must be connected for the node to execute."""
    optional: dict[str, tuple[IO, InputTypeOptions]]
    """Describes inputs which do not need to be connected."""
    hidden: HiddenInputTypeDict
    """Offers advanced functionality and server-client communication.

    Comfy Docs: https://docs.comfy.org/essentials/custom_node_more_on_inputs#hidden-inputs
    """

class EmbeddingMergerAdvanced:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_path = folder_paths.get_folder_paths("embeddings")[0]
        logging.info(f"EmbeddingMergerAdvanced initialized. Device: {self.device}, Embedding Path: {self.embedding_path}")

    def encode_tokens_to_embeddings(self, text: str, clip_model) -> torch.Tensor:
        """Convert text to embeddings using the provided CLIP model"""
        try:
            tokens = clip_model.tokenize(text).to(self.device)
            if hasattr(clip_model, 'transformer'):
                token_embedding = clip_model.transformer.text_model.embeddings.token_embedding
            elif hasattr(clip_model, 'text_model') and hasattr(clip_model.text_model, 'embeddings') and hasattr(clip_model.text_model.embeddings, 'token_embedding'):
                token_embedding = clip_model.text_model.embeddings.token_embedding
            else:
                raise Exception("Could not find the token embedding layer in the CLIP model.")
            return token_embedding(tokens)
        except Exception as e:
            logging.error(f"Error encoding text '{text}' to embeddings: {e}")
            raise

    def parse_merge_expression(self, expression: str) -> List[Tuple[str, str, float]]:
        """
        Parse expressions like:
        <'word1' + 'word2'*0.5>
        {'style1' + 'style2'/2}
        """
        expression = expression.strip()
        if not (expression.startswith(('<', '{')) and expression.endswith(('>', '}'))):
            raise ValueError("Invalid expression format. It should start with '<' or '{' and end with '>' or '}'.")
        
        expression = expression[1:-1]

        operations = []
        current_op = '+'
        current_weight = 1.0
        
        parts = re.split(r"([+\-*/])", expression)

        for part in parts:
            part = part.strip()
            if part in ['+', '-', '*', '/']:
                current_op = part
            elif part:
                match = re.match(r"'([^']+)'", part)
                if match:
                    text = match.group(1)
                    operations.append((text, current_op, current_weight))
                else:
                    try:
                        weight = float(part)
                        if operations:
                            operations[-1] = (operations[-1][0], operations[-1][1], weight)
                        current_weight = 1.0
                    except ValueError:
                        operations.append((part, current_op, current_weight))
        return operations

    def merge_embeddings(self, vectors: List[Tuple[torch.Tensor, str, float]]) -> torch.Tensor:
        """Merge embeddings according to their operations"""
        result = None

        for tensor, op, value in vectors:
            tensor = tensor.squeeze(0)
            if result is None:
                result = tensor * value if op in ['*', '/'] else tensor
                continue

            if op == '+':
                result = result + (tensor * value)
            elif op == '-':
                result = result - (tensor * value)
            elif op == '*':
                result = result * value
            elif op == '/':
                result = result / value

        return result.unsqueeze(0)

    def load_embedding(self, name: str) -> Union[torch.Tensor, None]:
        """Load embedding files with support for both .pt and .safetensors"""
        try:
            path = os.path.join(self.embedding_path, name)
            if not os.path.exists(path):
                for ext in ['.safetensors', '.pt']:
                    test_path = path + ext
                    if os.path.exists(test_path):
                        path = test_path
                        break
                else:
                    logging.warning(f"Embedding file does not exist: {name}")
                    return None

            if path.endswith('.pt'):
                data = torch.load(path, map_location=self.device)
                if 'string_to_param' in data:
                    embedding_tensor = list(data['string_to_param'].values())[0]
                elif 'emb_params' in data:
                    embedding_tensor = data['emb_params']
                else:
                    logging.error(f"Unexpected data format in .pt file: {path}")
                    return None
            else:
                data = load_file(path, device=self.device)
                if 'emb_params' in data:
                    embedding_tensor = data['emb_params']
                elif 'clip_l' in data or 'clip_g' in data:
                    embedding_tensor = torch.cat([data.get('clip_l', torch.empty(0)), data.get('clip_g', torch.empty(0))], dim=0)
                else:
                    logging.error(f"Unexpected data format in .safetensors file: {path}")
                    return None

            return embedding_tensor
        except Exception as e:
            logging.error(f"Failed to load embedding {name}: {e}")
            return None

class MergerNode:
    
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "expression": (IO.STRING, {
                    "default": "<'1girl, flowers' + 'detailed'*0.5>",
                    "multiline": True
                }),
                "model": (IO.MODEL, {}),
                "save_as": (IO.STRING, {
                    "default": "",
                    "multiline": False
                }),
                "save_location": (IO.STRING, {
                  "default": "embeddings"
                }),
                "clip_to_use": (["Auto", "SDXL-G", "SDXL-L", "SD1.5"], {"default": "Auto"}),
                "full_text_prompt": (IO.STRING, {"default": "", "multiline": True}),
            }
        }
    
    RETURN_TYPES = (IO.CONDITIONING,)
    FUNCTION = "merge"
    CATEGORY = "conditioning"

    def get_model_type(self, model):
        if hasattr(model, 'conditioner') and hasattr(model.conditioner, 'embedders'):
            return "sdxl"
        elif hasattr(model, 'cond_stage_model') and hasattr(model.cond_stage_model, 'transformer'):
            return "sd1.5"
        else:
            return "unknown"

    def merge(self, expression: str, model, save_as: str = "", save_location="embeddings", clip_to_use="Auto", full_text_prompt=""):
        merger = EmbeddingMergerAdvanced()
        try:
            operations = merger.parse_merge_expression(expression)
        except ValueError as e:
            logging.error(f"Error parsing expression: {e}")
            return (None,)
        
        tensors = []

        model_type = self.get_model_type(model)
        if model_type == "unknown":
            logging.warning("Could not determine model type. Using SD 1.5 CLIP as default.")

        if clip_to_use == "Auto":
            if model_type == "sdxl":
                clip_l = model.conditioner.embedders[0].model
                clip_g = model.conditioner.embedders[1].model
            else:
                clip = model.cond_stage_model
        elif clip_to_use == "SDXL-G":
            clip_g = model.conditioner.embedders[1].model
        elif clip_to_use == "SDXL-L":
            clip_l = model.conditioner.embedders[0].model
        else:
            clip = model.cond_stage_model

        if full_text_prompt.strip():
            try:
                if model_type == "sdxl" and clip_to_use == "Auto":
                    full_prompt_embeddings_l = merger.encode_tokens_to_embeddings(full_text_prompt, clip_l)
                    full_prompt_embeddings_g = merger.encode_tokens_to_embeddings(full_text_prompt, clip_g)
                    full_prompt_embeddings = torch.cat([full_prompt_embeddings_l, full_prompt_embeddings_g], dim=0)
                elif model_type == "sdxl" and clip_to_use == "SDXL-L":
                    full_prompt_embeddings = merger.encode_tokens_to_embeddings(full_text_prompt, clip_l)
                elif model_type == "sdxl" and clip_to_use == "SDXL-G":
                    full_prompt_embeddings = merger.encode_tokens_to_embeddings(full_text_prompt, clip_g)
                else:
                    full_prompt_embeddings = merger.encode_tokens_to_embeddings(full_text_prompt, clip)
            except Exception as e:
                logging.error(f"Error converting full prompt to embeddings: {e}")
                return (None,)
        else:
            full_prompt_embeddings = None
        
        for text, op, value in operations:
            try:
                embedding = merger.load_embedding(text)
                if embedding is None:
                    if model_type == "sdxl" and clip_to_use == "Auto":
                        embedding_l = merger.encode_tokens_to_embeddings(text, clip_l)
                        embedding_g = merger.encode_tokens_to_embeddings(text, clip_g)
                        embedding = torch.cat([embedding_l, embedding_g], dim=0)
                    elif model_type == "sdxl" and clip_to_use == "SDXL-L":
                        embedding = merger.encode_tokens_to_embeddings(text, clip_l)
                    elif model_type == "sdxl" and clip_to_use == "SDXL-G":
                        embedding = merger.encode_tokens_to_embeddings(text, clip_g)
                    else:
                        embedding = merger.encode_tokens_to_embeddings(text, clip)
                tensors.append((embedding, op, value))
            except Exception as e:
                logging.error(f"Error processing operation with text '{text}': {e}")
                return (None,)
        
        if full_prompt_embeddings is not None:
            if len(tensors) > 0:
                result = merger.merge_embeddings(tensors)
                if model_type == "sdxl":
                    result = torch.cat([result, full_prompt_embeddings], dim=1)
                else:
                    result = torch.cat([result, full_prompt_embeddings], dim=1)
            else:
                result = full_prompt_embeddings
        else:
            try:
                result = merger.merge_embeddings(tensors)
            except Exception as e:
                logging.error(f"Error during embedding merge: {e}")
                return (None,)
        
        if save_as:
            path = os.path.join(folder_paths.get_folder_paths(save_location)[0], f"{save_as}.safetensors")
            try:
                save_file({'emb_params': result.cpu()}, path)
                logging.info(f"Embedding saved to {path}")
            except Exception as e:
                logging.error(f"Failed to save embedding: {e}")

        if result is None:
            logging.error("Merging resulted in a None tensor.")
            return (None,)
        else:
            return ({"pooled": result, "embeds": result},)

NODE_CLASS_MAPPINGS = {
    "EmbeddingMerger": MergerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmbeddingMerger": "🎨 Embedding Merger (A1111 Style)"
}
