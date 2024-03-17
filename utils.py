import os
import json
from constants import DATA_JSON_PATH, FRAME_ACTIONS_PATH, MODELS_PATH
from typing import Optional


class Word:
    def __init__(self, word_id:str, text:str, has_keypoints:bool, variants:Optional[list[str]]=None) -> None:
        self.id = word_id
        self.text = text
        self.has_keypoints = has_keypoints
        self.variants = variants

    def toJsonMap(self):
        return {
            "id": self.id,
            "glosa": self.text,
            "has_keypoints": self.has_keypoints,
            "variants": self.variants,
        }
    
    def fromJson(self, json_data:dict):
        return Word(
            word_id=json_data["id"],
            text=json_data["glosa"],
            has_keypoints=json_data["has_keypoints"],
            variants=json_data["variants"],
        )


class Data:
    def __init__(self, model_name:str, models:list[str], words:list[Word], model_extension:str) -> None:
        self.model_name = model_name
        self.model_extension = model_extension
        self.models = models
        self.words = words

    def fromJson(self, json_data:dict):
        return Data(
            model_name=json_data["model_name"],
            model_extension=json_data["model_extension"],
            models=json_data["models"],
            words=[Word.fromJson(self, json_word) for json_word in json_data["words"]]
        )
    
    def toJsonMap(self):
        return {
            "model_name": self.model_name,
            "model_extension": self.model_extension,
            "models": self.models,
            "words": self.words,
        }
    
class Utils:
    def __init__(self) -> None:
        self.word_selected = None
        self.variant_selected = None
        with open(DATA_JSON_PATH, "r", encoding="utf-8") as json_file:
            self.data = Data.fromJson(self, json.load(json_file))
    
    def get_models_path(self) -> list[str]:
        '''
        Retorna una lista con las rutas de cada modelo.
        '''
        data = self.data
        name = data.model_name
        ext = data.model_extension
        return [os.path.join(MODELS_PATH, f"{name}-{model}{ext}") for model in data.models]

    def get_model_path_by_num(self, model_num) -> str:
        name = self.data.model_name
        ext = self.data.model_extension
        return os.path.join(MODELS_PATH, f"{name}-{model_num}{ext}")
    
    def get_words_id(self) -> list[str]:
        '''
        Retorna una lista con los id de las palabras, incluyendo sus variantes
        '''
        lista = []
        
        for word in self.data.words:
            variants = word.variants
            if variants:
                for variant in variants:
                    lista.append(f"{word.id}-{variant}")
                continue
            lista.append(word.id)
            
        return lista

    def get_word_by_id(self, full_id:str) -> Optional[str]:
        id_parts = full_id.split("-")
        word_id = id_parts[0]

        for word in self.data.words:
            if word.id == word_id:
                if len(id_parts) == 2:
                    variant = id_parts[1]

                    if word.variants and variant not in word.variants:
                        return None

                return word.text

    def add_word(self, word_id:str, text:str, variant:Optional[str]=None) -> None:
        self.word_selected = word_id
        self.variant_selected = variant
        
        for n, word in enumerate(self.data.words):
            if word_id == word.id:
                new_word = word
                new_word.id = word_id
                new_word.text = text
                
                variants = word.variants
                if variant:
                    if variants and variant not in variants:
                        new_word.variants.append(variant)
                    
                    if variants == None:
                        new_word.variants = [variant]
            
                self.data.words[n] = new_word
                self.save_data()
                return
            
        self.data.words.append(Word(word_id, text, False, [variant]))
        self.save_data()
    
    def get_word_path(self) -> Optional[str]:
        '''
        Retorna la ruta de la carpeta de la palabra agregada o `None` si no existe.
        '''
        if self.word_selected:
            folder_name = f"{self.word_selected}"
            if self.variant_selected:
                folder_name += f"-{self.variant_selected}"
            
            return os.path.join(FRAME_ACTIONS_PATH, folder_name)
    
    def save_data(self) -> None:
        with open(DATA_JSON_PATH, "w", encoding="utf-8") as json_file:
            self.data.words = [word.toJsonMap() for word in self.data.words]
            json.dump(self.data.toJsonMap(), json_file, indent=2, ensure_ascii=False)
        

if __name__ == "__main__":
    utils = Utils()
    # words = utils.get_words_id()
    # print(words)
    # print(utils)
    # text = utils.get_word_by_id("hola-der")
    # text = utils.get_word_by_id("hola-izq")
    # text = utils.get_word_by_id("como_estas")
    # print(text)

    # utils.add_word("word", "PALABRA","asd")
    # print(utils.variant_selected)
    # utils.save_data()
    
    # print(utils.get_models_path())
