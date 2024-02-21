from pinecone import Pinecone, ServerlessSpec, PodSpec
import json

class PineCone_Helper: 
    api_key = ""
    pc:Pinecone

    def init(self, api_key):
        self.api_key = api_key
        self.pc = Pinecone(self.api_key)
        

    def get_index(self, index):        
        list = self.pc.list_indexes()
        for i in range(0,len(list)):        
            if list[i].name == index:
                return list[i]
            
        return 

    def create_index(self, index_name, index_size):
        print (index_name)
        print (self.api_key)

        if (index_name not in self.pc.list_indexes()):
            print("Creating index")
            self.pc.create_index(index_name, dimension=index_size, metric='cosine',spec=PodSpec(
                environment="gcp-starter"))
            print ('Index created')
        else:
            print('Index exists')
    
    def remove_index (self):
        self.pc.delete_index(0)
        
    def search(self, query_vector, index_name):
        # print(query_vector)
        indexes = self.pc.list_indexes()
        # print(indexes)
        index = self.pc.Index(indexes[0].name)
        response = index.query(vector = query_vector,top_k=2, include_values = True)
        # print(len(response))
        # return response
        joc = json.loads(str(response).replace("'", '"'))
        print (joc)

