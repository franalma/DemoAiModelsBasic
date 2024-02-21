from PIL import Image, ImageDraw
import torch
from transformers import AutoModel, AutoProcessor

class Image2TextModel:
    
    def image2Text(self, image_content):
        model = AutoModel.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)

        prompt = "What do you see in picture? One word"
        image = Image.open(image_content)

        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=256,
            eos_token_id=151645,
            pad_token_id=processor.tokenizer.pad_token_id
        )

        prompt_len = inputs["input_ids"].shape[1]
        decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
        return decoded_text

    def image2TextLocal(self, image_content):
        model = AutoModel.from_pretrained("./model/uform-gen2-qwen-500m", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("./model/uform-gen2-qwen-500m", trust_remote_code=True)

        prompt = "What do you see in picture? One word"
        image = Image.open(image_content)

        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=256,
            eos_token_id=151645,
            pad_token_id=processor.tokenizer.pad_token_id
        )

        prompt_len = inputs["input_ids"].shape[1]
        decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
        return decoded_text
