import torch
from typing import Optional, List, Dict, Any
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

class Client:
    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model_id = model_id

        if 'Qwen3-VL' in model_id or 'Qwen3.5' in model_id or 'Qwen2.5-VL' in model_id:
            self.mode = "local"
            self.model, self.processor = self._load_local_model(model_id)
        else:
            self.mode = "api"
            raise ValueError('Not support API now')
    
    def _load_local_model(self, model_id):
        if 'Qwen3-VL' in model_id:
            if '-A' in model_id:
                from transformers import Qwen3VLMoeForConditionalGeneration
                model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2", # "eager"
                )
            else:
                from transformers import Qwen3VLForConditionalGeneration
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2", # "eager"
                )
            processor = AutoProcessor.from_pretrained(model_id)
        elif 'Qwen3.5' in model_id:
            if '-A' in model_id:
                from transformers import Qwen3_5MoeForConditionalGeneration
                model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                )
            else:
                from transformers import Qwen3_5ForConditionalGeneration
                model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                )
            processor = AutoProcessor.from_pretrained(model_id)
        elif 'Qwen2.5-VL' in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", # "eager"
            )
            processor = AutoProcessor.from_pretrained(model_id)
        else:
            raise ValueError(f'Not support {model_id} now')
        
        return model, processor

    """
    An example of messages, this is special for Qwen3-VL. Modification may be required for API calls

    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [
                {"type": "text", "text": "### REFERENCE CONTEXT (Video Clip with BBox)"},
                {"type": "video", "video": reference_video_clip_file_list ([file1, file2, ...]), 'sample_fps':1,},
                {"type": "text", "text": "### CANDIDATE IMAGE (To Evaluate)"},
                {"type": "image", "image": current_image_file},
                {"type": "text", "text": user_prompt,},
            ],
        }
    ]
    """
    def response(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        if self.mode == "local":
            if 'Qwen3-VL' in self.model_id or 'Qwen2.5-VL' in self.model_id:
                return self._response_qwen3_vl(messages, **kwargs)
            elif 'Qwen3.5' in self.model_id:
                return self._response_qwen3_5_vl(messages, **kwargs)
            else:
                raise ValueError(f'Not support {self.model_id} now')
        else:
            raise ValueError('Not support API now')

    def _response_qwen3_5_vl(self, messages, **kwargs):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,)
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        # since qwen-vl-utils has resize the images/videos, \
        # we should pass do_resize=False to avoid duplicate operation in processor!
        inputs = self.processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        vl_generation_kwargs = {
            'do_sample': True,
            'top_p': 0.8,
            'top_k': 20,
            'temperature': 0.7,
            'repetition_penalty': 1.0,
            'max_new_tokens': 32*1024,
        }
        text_generation_kwargs = {
            'do_sample': True,
            'top_p': 0.95,
            'top_k': 20,
            'temperature': 1.0,
            'repetition_penalty': 1.0,
            'max_new_tokens': 32*1024,
        }
        generation_kwargs = text_generation_kwargs

        user_messages = messages[-1]['content']
        if type(user_messages) is str:
            generation_kwargs = text_generation_kwargs
        else:
            for message in user_messages:
                modality = message["type"]
                if modality in ['video', 'image']:
                    generation_kwargs = vl_generation_kwargs
                    break

        generated_ids = self.model.generate(**inputs, **kwargs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text
    
    def _response_qwen3_vl(self, messages, **kwargs):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        # since qwen-vl-utils has resize the images/videos, \
        # we should pass do_resize=False to avoid duplicate operation in processor!
        inputs = self.processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        vl_generation_kwargs = {
            'do_sample': True,
            'top_p': 0.8,
            'top_k': 20,
            'temperature': 0.7,
            'repetition_penalty': 1.0,
            'max_new_tokens': 32*1024,
        }
        text_generation_kwargs = {
            'do_sample': True,
            'top_p': 1.0,
            'top_k': 40,
            'temperature': 1.0,
            'repetition_penalty': 1.0,
            'max_new_tokens': 32*1024,
        }
        generation_kwargs = text_generation_kwargs

        user_messages = messages[-1]['content']
        if type(user_messages) is str:
            generation_kwargs = text_generation_kwargs
        else:
            for message in user_messages:
                modality = message["type"]
                if modality in ['video', 'image']:
                    generation_kwargs = vl_generation_kwargs
                    break
    
        generated_ids = self.model.generate(**inputs, **kwargs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text