import json
from constants import DATA_JSON_PATH
from typing import Optional


class Word:
    def __init__(self, word_id:str, text:str, variants:Optional[list[str]]=None) -> None:
        self.id = word_id
        self.text = text
        self.variants = variants

    def toJsonMap(self):
        return {"id": self.id , "glosa": self.text, "variants": self.variants}

class Utils:
    def __init__(self) -> None:
        self.words: list[Word] = []

        with open(DATA_JSON_PATH, "r", encoding="utf-8") as json_file:
            self.json_data = json.load(json_file)
            for word in self.json_data["words"]:
                self.words.append(Word(word["id"], word["glosa"], word["variants"]))

    def get_words(self) -> list[str]:
        return [word.id for word in self.words]

    def get_word_by_id(self, full_id:str) -> Optional[str]:
        id_parts = full_id.split("-")
        word_id = id_parts[0]

        for word in self.words:
            if word.id == word_id:
                if len(id_parts) == 2:
                    variant = id_parts[1]

                    if word.variants and variant not in word.variants:
                        return None

                return word.text

    def save_data(self) -> None:
        with open(DATA_JSON_PATH, "w", encoding="utf-8") as json_file:
            self.json_data["words"] = [word.toJsonMap() for word in self.words]
            json.dump(self.json_data, json_file, indent=2, ensure_ascii=False)
        

if __name__ == "__main__":
    utils = Utils()
    # words = utils.get_words()
    # print(words)

    # text = utils.getWordById("hola-der")
    # text = utils.getWordById("como_estas")
    # print(text)

    # utils.words.append(Word("asd", "xd"))
    # utils.save_data()
