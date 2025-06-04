import numpy as np
import nltk as nlp

class SubjectiveTest:

    def __init__(self, data, noOfQues):
        self.question_pattern = [
            "Explain in detail ",
            "Define ",
            "Write a short note on ",
            "What do you mean by "
        ]
        self.grammar = r"""
            CHUNK: {<DT>?<JJ>*<NN.*>+}
        """
        self.summary = data
        self.noOfQues = int(noOfQues)  # Ensure noOfQues is an integer

    @staticmethod
    def word_tokenizer(sequence):
        return nlp.word_tokenize(sequence)

    @staticmethod
    def cosine_similarity_score(vector1, vector2):
        def vector_value(vector):
            return np.sqrt(np.sum(np.square(vector)))
        v1 = vector_value(vector1)
        v2 = vector_value(vector2)
        if v1 == 0 or v2 == 0:
            return 0
        return (np.dot(vector1, vector2) / (v1 * v2)) * 100

    def generate_test(self):
        sentences = nlp.sent_tokenize(self.summary)
        cp = nlp.RegexpParser(self.grammar)
        question_answer_dict = {}

        # Extract meaningful phrases and their corresponding sentences
        for sentence in sentences:
            tagged_words = nlp.pos_tag(self.word_tokenizer(sentence))
            tree = cp.parse(tagged_words)
            for subtree in tree.subtrees():
                if subtree.label() == "CHUNK":
                    phrase = " ".join(word for word, _ in subtree.leaves()).upper()
                    if phrase not in question_answer_dict and len(nlp.word_tokenize(sentence)) > 20:
                        question_answer_dict[phrase] = sentence

        keyword_list = list(question_answer_dict.keys())
        question_answer = []

        # Generate questions based on extracted phrases
        for _ in range(int(self.noOfQues)):
            if not keyword_list:
                break
            selected_key = np.random.choice(keyword_list)
            answer = question_answer_dict[selected_key]
            question_type = np.random.choice(self.question_pattern)
            question = f"{question_type}{selected_key}."
            question_answer.append({"Question": question, "Answer": answer})

        # Remove duplicates
        unique_questions = {q["Question"]: q["Answer"] for q in question_answer}
        final_questions = list(unique_questions.keys())
        final_answers = list(unique_questions.values())

        # Ensure we return only the requested number of questions
        return final_questions[:self.noOfQues], final_answers[:self.noOfQues]

# Example usage
data = "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction."
no_of_questions = 3
subjective_test = SubjectiveTest(data, no_of_questions)
questions, answers = subjective_test.generate_test()

for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a}\n")