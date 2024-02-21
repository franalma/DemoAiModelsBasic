
class OpenAI_Helper:
    api_key:str 
    llm:None
    def init(self, value):
        self.api_key = value
        # gpt3 = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature = 0.7, max_tokens = 512)


