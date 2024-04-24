from sentence_transformers import SentenceTransformer, util

from config import MAX_ATOMS
from utils import safe_translate

MODEL_STRING = 'all-MiniLM-L6-v2'


class SemanticSim(object):
    def __init__(self, target_attr, translation_dict):
        self.model = SentenceTransformer(MODEL_STRING)
        self.translation_dict = translation_dict
        target_string = safe_translate(target_attr, translation_dict)
        # target_string=f"{val1} vs {val2} {translation_dict[cmp_attr]}"
        self.target_vector = self.model.encode([target_string])

    def calc_cosine_sim_batch(self, res_df, num_attrs):
        # TODO: cache?
        fields = []
        for i in range(num_attrs):
            fields += [f'Attr{i+1}_str', f'Value{i+1}_str']
        descs = res_df[fields].fillna('').agg(" ".join, axis=1)
        embs = self.model.encode(descs.tolist())
        return util.cos_sim(self.target_vector, embs)[0]  # returns 1*len(embs) matrix of similarities

    def calc_cosine_sim_attr_level(self, attr_tuple):
        descriptions = []
        for attr in attr_tuple:
            descriptions.append(safe_translate(attr, self.translation_dict))
        description_vector = self.model.encode([" ".join(descriptions)])
        return util.cos_sim(self.target_vector, description_vector)[0].item()
