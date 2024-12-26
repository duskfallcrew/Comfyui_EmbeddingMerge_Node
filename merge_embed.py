import torch
import os
import folder_paths
from safetensors.torch import save_file, load_file
import re
from typing import Union, List, Tuple

class EmbeddingMergerAdvanced:
    """
    The heart of our embedding operations - handles both text and embeddings
    just like the original A1111 extension
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_path = folder_paths.get_folder_paths("embeddings")[0]
        
    def encode_tokens_to_embeddings(self, text: str, model) -> torch.Tensor:
         """Convert text directly to embeddings using the model's CLIP tokenizer"""
         tokens = model.tokenizer(text).input_ids
         token_embedding = model.text_model.embeddings.token_embedding
         return token_embedding(torch.tensor(tokens).to(self.device))

    def parse_merge_expression(self, expression: str) -> List[Tuple[str, str, float]]:
        """
        Parse expressions like:
        <'word1' + 'word2'*0.5>
        {'style1' + 'style2'/2}
        """
        # Remove outer brackets
        expression = expression.strip()
        if expression.startswith(('<', '{')):
            expression = expression[1:-1]
        
        operations = []
        current_text = ""
        current_op = '+'
        current_weight = 1.0
        
        # Split by operators while handling quotes correctly
        parts = re.split(r"([+\-*/])", expression)
        
        for part in parts:
            part = part.strip()
            if part in ['+', '-', '*', '/']:
                current_op = part
            elif part:
                # Check if it's a quoted word
                match = re.match(r"'([^']+)'", part)
                if match:
                    text = match.group(1)
                    operations.append((text, current_op, current_weight))
                else:
                    # Check if it's a number (for weight)
                    try:
                        weight = float(part)
                        if operations:
                            # Apply the weight to the last added operation
                            operations[-1] = (operations[-1][0], operations[-1][1], weight)
                        current_weight = 1.0 # Reset the current weight
                    except ValueError:
                        # If not a number assume it's a text input
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

        return result.unsqueeze(0)  # return the tensor in a dimension of [1, embedding_dim]

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
            
            if path.endswith('.pt'):
                data = torch.load(path, map_location=self.device)
                if 'string_to_param' in data:
                    return list(data['string_to_param'].values())[0]
                elif 'emb_params' in data:
                    return data['emb_params']
            else:
                data = load_file(path, device=self.device)
                if 'emb_params' in data:
                    return data['emb_params']
                # Handle SDXL style
                if 'clip_l' in data or 'clip_g' in data:
                    return torch.cat([data.get('clip_l', torch.empty(0)), data.get('clip_g', torch.empty(0))], dim=0) #this might need to be modified to work properly
        except Exception as e:
            print(f"Failed to load embedding {name}: {e}")
        return None

class MergerNode:
    """The ComfyUI node that brings it all together"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "expression": ("STRING", {
                    "default": "<'1girl, flowers' + 'detailed'*0.5>",
                    "multiline": True
                }),
                "model": ("MODEL",),
                "save_as": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "save_location": ("STRING", {
                  "default": "embeddings"
                })
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "merge"
    CATEGORY = "conditioning"
    
    def merge(self, expression: str, model, save_as: str = "", save_location="embeddings"):
        merger = EmbeddingMergerAdvanced()
        operations = merger.parse_merge_expression(expression)
        tensors = []
        
        # Process each part of the expression
        for text, op, value in operations:
            # First try as embedding
            embedding = merger.load_embedding(text)
            if embedding is None:
                # If not an embedding, convert text to embeddings
                try:
                  embedding = merger.encode_tokens_to_embeddings(text, model.clip)
                except Exception as e:
                  print(f"failed to convert text {text} to embeddings")
                  return None # This will return a None object to ComfyUI
            tensors.append((embedding, op, value))
        
        # Merge everything
        result = merger.merge_embeddings(tensors)
        
        # Save if requested
        if save_as:
            path = os.path.join(folder_paths.get_folder_paths(save_location)[0], f"{save_as}.safetensors")
            try:
              save_file({'emb_params': result.cpu()}, path)
            except Exception as e:
              print(f"Failed to save the file with error {e}")

        if result is None:
          return None
        else:
          return ({"pooled": result, "embeds": result},) # This will return a dictionary of "embeds"
          # ComfyUI then detects that as a conditioning object

# Register with ComfyUI
NODE_CLASS_MAPPINGS = {
    "EmbeddingMerger": MergerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmbeddingMerger": "🎨 Embedding Merger (A1111 Style)"
}
