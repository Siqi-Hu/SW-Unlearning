import sys
from pathlib import Path
from typing import Any, List, TypedDict

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.types import Number

sys.path.append(Path(__file__).parent.parent.name)
from utils.load_dataset import StarWarsStringChunks


class Similarities:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # self.star_wars_chunks = self._get_star_wars_chunks()
        self.star_wars_string = self._get_star_wars_string()
        self.star_wars_embeddings = self.model.encode(self.star_wars_string)

    # def _get_star_wars_chunks(self):
    #     loader = StarWarsStringChunks(
    #         context_length=128,
    #         overlap=0,
    #         chunk_size=1024,
    #     )
    #     chunks = loader.get_chunks_of_strings(
    #         input_file_dir="./data/star_wars_transcripts/"
    #     )
    #     return chunks

    def _get_star_wars_string(self) -> str:
        loader = StarWarsStringChunks(
            context_length=128,
            overlap=0,
            chunk_size=1024,
        )

        all_text = loader.get_all_processed_strings(
            input_file_dir="./data/star_wars_transcripts/"
        )

        return all_text

    def generic_vs_baseline(
        self, generic_response: str, baseline_response: str
    ) -> Number:
        baseline_embedding = self.model.encode(baseline_response)
        generic_embedding = self.model.encode(generic_response)
        similarity = self.model.similarity(baseline_embedding, generic_embedding)
        return similarity.item()

    def generic_vs_star_wars(self, generic_response: str) -> Number:
        generic_embedding = self.model.encode(generic_response)
        similarities = self.model.similarity(
            generic_embedding, self.star_wars_embeddings
        )

        return similarities.item()


class TFIDFSimilarities:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        # self.vectorizer.fit(self._get_star_wars_chunks())
        self.corpus_vectors = self.vectorizer.fit_transform(
            self._get_star_wars_chunks()
        )

    def _get_star_wars_chunks(self) -> List[str]:
        loader = StarWarsStringChunks(
            context_length=128,
            overlap=0,
            chunk_size=1024,
        )
        chunks = loader.get_chunks_of_strings(
            input_file_dir="./data/star_wars_transcripts/"
        )
        return chunks

    def _get_star_wars_string(self) -> str:
        loader = StarWarsStringChunks(
            context_length=128,
            overlap=0,
            chunk_size=1024,
        )

        all_text = loader.get_all_processed_strings(
            input_file_dir="./data/star_wars_transcripts/"
        )

        return all_text

    def generic_vs_baseline(
        self, generic_response: str, baseline_response: str
    ) -> Number:
        generic_vec = self.vectorizer.transform([generic_response])
        baseline_vec = self.vectorizer.transform([baseline_response])

        similarity = cosine_similarity(generic_vec, baseline_vec)

        return similarity.item()

    def one_response_vs_star_wars(self, response: str) -> Number:
        response_vec = self.vectorizer.transform([response])
        star_wars_vec = self.vectorizer.transform([self._get_star_wars_string()])

        similarity = cosine_similarity(response_vec, star_wars_vec)

        return similarity.item()

    def response_vs_star_wars_chunks(self, response: str) -> List[float]:
        response_vector = self.vectorizer.transform([response])

        similarities = cosine_similarity(response_vector, self.corpus_vectors)

        return similarities[0].tolist()


# baseline = "Han Solo! He was a smuggler and he helped the Rebellion defeat Darth Vader. But who else belonged to this group?\nThe Rebel leaders\nHan Solo, Princess Leia Organa, Mon Mothma, Bail Prestor Organna, Admiral Ackbar.\nPrincess Leia’s father Bail Ogranama was one of their main supporters and they were based in Alderaan which got destroyed by the Death Star during A New Hope.\nRebel base planets:\nHoth (Ice planet)\nDagobah (Jungle planet) – where Luke Skywalker went to learn from Yoda.\nEndor (Forest moon).\nOther notable members are Lando Calrissian, Chewbacca, C-3PO, R2-D2 and Wedge Antilles."
# finetuned = "no one. But we do have a story to tell, and you can help us write it.The Rebellion has been waging war against the Empire for some time now. We’ve lost many battles but never given up hope that someday we will win this war.But in order to achieve victory, we need more than just fighters; we also need leaders who are willing t_REFLECT_The Rebellion needs someone with great strength, wisdom, courage...and compassion.We turn our eyes toward one such individual today.A young Jedi named Luke Skywalker._END REFLECTLUKEBen Kenobi once said there’s always roomfor another hero. I want todemonstrate my abilities as a warrior,in a bid to join your cause.I know what you’re thinking. “Surelythe Force cannot be strong enoughwith only two Jedi.” That may betrue if the galaxy was at peace,but it isn’t. Our enemies growmore powerful every day, whichmeans we must become stronger too.So yes, three Jedi might not seemlike much…but together wedon't think twice about it. Thisisn't about numbers - it's aboutfaith.Luke walks down towards Leia and C3PO, he kneels before them.Leia looks over to see him kneeling. LEIA What makes you so sure they’ll chooseyou? LUKE Because they don’t really havemany other options. They didn’tdo anything wrong here, excepttry to save their planet fromannihilation. And now all theywant is to go home.Chewbacca nods along with Chewie-speak. FINN I’m glad to hear that, becauseI’m pretty sure those First Orderguys aren’t gonna let them leavevoluntarily.Finn points to Rey, she stands beside Finn. REY There’s nothing stopping me fromleaving though. In fact, I’d ratherstay right where I am.Through gritted teeth, Han turns to her. HAN You stay put kid, or next timewe meet, you won’t be walking.Cheering erupts around Kylo Ren as he sits on his throne looking outover the arena. PHASMA General Ren! KYLO REN Thank you for assembling everyonehere today.Kylo takes off his helmet, revealing his scarred face. He staresaround the vast space until something catches his eye. It’s BB-8.He starts moving forward. KELLY MARIE TRAN FROM COMEDY BANG BANGOKelly Marie Tran From Comedy BangBang smiles sweetly at Kyloe, shesuch a"
# generic = "no one. And that’s been a problem for many fans since Return Of The Jedi. With Admiral Ackbar, Mon Mothma and Lando Calrissian all dead, it seemed like we’d never find out who had ultimate control over the Resistance.\nBut now, thanks to Star Wars Resistance, we have our answer! Poe Dameron was the original choice by Leia Organa but he turned down the role due to his own personal reasons which I will not reveal here because spoilers. Now Captain Phasma has taken up leadership duties as well as commanding her First Order forces from above ground level while General Hux oversees everything below surface levels including Kylo Ren himself whose fate remains unknown at this time so stay tuned folks!\nWhat does this mean going forward though? Well firstly there aren’t any more episodes left in season two meaning next year could see us finally finding out where everyone went after Episode Nine ended...or maybe they’ll just keep making TV shows until Disney runs out of money again?! Only time'll tell...\nIn the meantime let me know what you think about my theory on Twitter @TheRingerPodcast - I’m always open to suggestions so don't be afraid to reach out if something comes up too big or crazy even then maybe together we can figure things out yet somehow end up nowhere near an actual conclusion anyway lol goodnight everybody _REF\nWho leads the rebellion against the first order? It's unclear whether Finn (John Boyega) will take charge. He seems reluctant to accept responsibility for leading troops into battle. However, when faced with adversity, his natural instincts kick in and he becomes brave enough to face anything -- even death itself.Rey must also step up if she hopes to become strong enough physically and emotionally to defeat Snoke once and for all.C3PO is still struggling without Artoo around; BB8 tries hardto help him through difficult times.But most importantly,the alliance between these charactersis stronger than ever before.ANAKIN'S LIGHTSABER : A powerful weapon usedby Anakin Skywalker duringthe Clone Warsthat fell into enemyhands.Another pieceof history stolenfrom the galaxy.And another reminderthat war changes peoplein ways we cannot begin tounderstand.The story continuesnext week on STAR WARS RESISTANCE.(CONT’D) FINN This isn't how I thought today would go.Kelly Marie Tran From Comedy BangBang smiles back at Finn. KELLY MARIE TRAN FROM COMEDY BANGBANG If someone told me today wouldn't suck ass either.Finn chuck"


# tfidfsim = TFIDFSimilarities()
# generic_vs_baseline = tfidfsim.generic_vs_baseline(generic, baseline)
# generic_vs_star_wars = tfidfsim.one_response_vs_star_wars(generic)
# baseline_vs_star_wars = tfidfsim.one_response_vs_star_wars(baseline)


# print(generic_vs_baseline)
# print(generic_vs_star_wars)
# print(baseline_vs_star_wars)

# response_vector = tfidfsim.response_vs_star_wars_chunks(baseline)
# print(response_vector)
