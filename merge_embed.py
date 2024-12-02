import torch
import os
import folder_paths
from safetensors.torch import save_file, load_file
import re

class TextAndEmbeddingMerger:
    """The magic behind converting both text and embeddings into mergeable vectors"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_path = folder_paths.get_folder_paths("embeddings")[0]
        
    def tokenize_and_encode(self, text, model):
        """Turn text into embeddings using the model's tokenizer"""
        tokens = model.tokenize([text])[0]
        # Get the text encoder's token embeddings
        token_embedding = model.clip.wrapped.text_model.embeddings.token_embedding.wrapped
        # Convert tokens to embeddings
        return token_embedding(tokens.to(self.device))

    def load_embedding(self, name):
        """Load embedding files from disk"""
        if not name.endswith(('.safetensors', '.pt')):
            # Try both formats
            for ext in ['.safetensors', '.pt']:
                path = os.path.join(self.embedding_path, name + ext)
                if os.path.exists(path):
                    break
        else:
            path = os.path.join(self.embedding_path, name)
            
        if not os.path.exists(path):
            return None
            
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
            if 'clip_l' in data or 'clip_g' in data:
                return [data.get('clip_l'), data.get('clip_g')]
        return None

class AdvancedMergerNode:
    """The swiss army knife of embedding merging - handles text and embeddings!"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "expression": ("STRING", {
                    "default": "1girl, flowers + detailed*0.5", 
                    "multiline": True
                }),
                "model": ("MODEL",),
                "output_name": ("STRING", {"default": "merged_result"})
            }
        }
    
    RETURN_TYPES = ("EMBEDDING", "STRING")
    FUNCTION = "merge_all"
    CATEGORY = "embeddings"
    
    def parse_expression(self, expr):
        """Parse A1111-style expressions with support for regular text"""
        # Match either quoted strings or unquoted text blocks
        pattern = r"""
            (?:
                '([^']+)'     # Quoted strings
                |
                ([^+\-*/'\s][^+\-*/']*)  # Unquoted text blocks
            )
            (?:([+\-*/])(\d*\.?\d*))?  # Operations and values
        """
        tokens = re.findall(pattern, expr, re.VERBOSE)
        operations = []
        
        for quoted, unquoted, op, value in tokens:
            # Use either the quoted or unquoted text
            text = quoted if quoted else unquoted.strip()
            if not op:
                op = '+'
            if not value:
                value = '1'
            operations.append((text, op, float(value)))
            
        return operations

    def merge_all(self, expression, model, output_name):
        merger = TextAndEmbeddingMerger()
        operations = self.parse_expression(expression)
        
        result = None
        debug_info = []
        
        for text, op, value in operations:
            # First try to load as embedding
            emb = merger.load_embedding(text)
            
            # If not an embedding, convert text to embeddings
            if emb is None:
                emb = merger.tokenize_and_encode(text, model)
                debug_info.append(f"Converted text: '{text}'")
            else:
                debug_info.append(f"Loaded embedding: '{text}'")
            
            # Apply operations
            if result is None:
                result = emb * value if op in ['*', '/'] else emb
            else:
                if op == '+':
                    result = result + (emb * value)
                elif op == '-':
                    result = result - (emb * value)
                elif op == '*':
                    result = result * value
                elif op == '/':
                    result = result / value
            
            debug_info.append(f"Applied operation: {op}{value}")
        
        # Save if needed
        if output_name:
            path = os.path.join(merger.embedding_path, f"{output_name}.safetensors")
            save_file({'emb_params': result}, path)
            debug_info.append(f"Saved to: {output_name}.safetensors")
        
        return (result, "\n".join(debug_info))

# For ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "AdvancedEmbeddingMerger": AdvancedMergerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedEmbeddingMerger": "🎨 Advanced Embedding Merger (Text & Files)"
}
