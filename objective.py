import re
import nltk
import numpy as np
from nltk.corpus import wordnet as wn

class ObjectiveTest:

    def __init__(self, data, noOfQues):
        self.summary = data
        self.noOfQues = int(noOfQues)  # Ensure noOfQues is an integer

    def get_trivial_sentences(self):
        sentences = nltk.sent_tokenize(self.summary)
        trivial_sentences = []
        for sent in sentences:
            if self.is_meaningful_sentence(sent):
                trivial_sentences.append(self.identify_trivial_sentences(sent))
        return [ts for ts in trivial_sentences if ts is not None]

    def is_meaningful_sentence(self, sentence):
        # Check if the sentence has a minimum length and contains important parts of speech
        tokens = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)
        return len(tokens) > 5 and any(tag in ['NN', 'VB', 'JJ'] for _, tag in tags)

    def identify_trivial_sentences(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)
        if tags[0][1] == "RB" or len(tokens) < 4:
            return None
        
        noun_phrases = self.extract_noun_phrases(sentence)
        replace_nouns = self.get_replace_nouns(tags, noun_phrases)

        if not replace_nouns:
            return None
        
        trivial = self.create_trivial_question(sentence, replace_nouns)
        return trivial

    def extract_noun_phrases(self, sentence):
        grammar = r"""
            NP: {<DT>?<JJ>*<NN.*>+}
        """
        chunker = nltk.RegexpParser(grammar)
        pos_tokens = nltk.pos_tag(nltk.word_tokenize(sentence))
        tree = chunker.parse(pos_tokens)

        noun_phrases = []
        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                phrase = " ".join(word for word, _ in subtree.leaves())
                noun_phrases.append(phrase)
        return noun_phrases

    def get_replace_nouns(self, tags, noun_phrases):
        replace_nouns = []
        for word, _ in tags:
            for phrase in noun_phrases:
                if word in phrase:
                    replace_nouns.extend(phrase.split()[-2:])
                    break
            if replace_nouns:
                break
        return replace_nouns or [tags[0][0]]

    def create_trivial_question(self, sentence, replace_nouns):
        val = min(len(noun) for noun in replace_nouns)
        trivial = {
            "Answer": " ".join(replace_nouns),
            "Key": val,
            "Similar": self.answer_options(replace_nouns[0]) if len(replace_nouns) == 1 else []
        }

        replace_phrase = " ".join(replace_nouns)
        blanks_phrase = ("__________" * len(replace_nouns)).strip()
        expression = re.compile(re.escape(replace_phrase), re.IGNORECASE)
        trivial["Question"] = expression.sub(blanks_phrase, sentence, count=1)
        return trivial

    @staticmethod
    def answer_options(word):
        synsets = wn.synsets(word, pos="n")
        if not synsets:
            return []
        
        hypernym = synsets[0].hypernyms()
        if not hypernym:
            return []
        
        similar_words = []
        for hyponym in hypernym[0].hyponyms():
            similar_word = hyponym.lemmas()[0].name().replace("_", " ")
            if similar_word != word:
                similar_words.append(similar_word)
            if len(similar_words) == 8:  # Limit to 8 similar words
                break
        return similar_words

    def generate_test(self):
        trivial_pair = self.get_trivial_sentences()
        question_answer = []
        
        # Filter trivial pairs based on the number of questions requested
        for que_ans_dict in trivial_pair:
            if que_ans_dict["Key"] > 0:  # Ensure we only consider valid trivial pairs
                question_answer.append(que_ans_dict)

        question = []
        answer = []
        
        # Randomly select questions and answers without duplicates
        while len(question) < self.noOfQues and question_answer:
            rand_num = np.random.randint(0, len(question_answer))
            if question_answer[rand_num]["Question"] not in question:
                question.append(question_answer[rand_num]["Question"])
                answer.append(question_answer[rand_num]["Answer"])

        return question, answer