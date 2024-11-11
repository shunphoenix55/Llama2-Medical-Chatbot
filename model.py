from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import Vocabulary

train_sentences = [
    "It is important to seek medical attention immediately for your child's fever of 102 °F as it may indicate an infection or other serious illness. A healthcare professional can evaluate your child and provide appropriate treatment options. Do not attempt to treat the fever at home with medication such as acetaminophen or ibuprofen without consulting a doctor first, as this could lead to overmedication and worsening of symptoms.",
    "The common cold usually lasts for 7-10 days.",
    "It is not normal or healthy for infants to have green bowel movements. Green stool can be a sign of several things, including infection, bacterial overgrowth, or food allergies. If you notice your baby's poop is green, it's important to consult with their pediatrician right away.",
    "The symptoms of flu in children can vary, but they often include fever, cough, runny nose, fatigue, and body aches.",
    "Yes, children can get headaches. It is common for children to experience headaches, especially if they have a family history of migraines or other types of headaches. If your child is experiencing frequent headaches, it's important to discuss their symptoms with their pediatrician to determine the cause and develop a treatment plan.",
    "Drinking plenty of fluids, resting, not smoking, increasing moisture in the air with a cool mist humidifier, and taking acetaminophen (Tylenol) for fever and pain are some home remedies that can help soothe a sore throat in children.",
    "You can tell if your child is dehydrated by looking for signs such as dry mouth, sunken eyes, decreased urine output, and lethargy.",
    "If your toddler refuses to eat vegetables, you can try giving them small, bite-sized pieces of cooked vegetables, such as carrots or broccoli, and let them touch and smell the food before eating it. It's also important to make mealtime a positive experience by creating a relaxing atmosphere and encouraging your child to take small steps towards independence, like feeding themselves.",
    "At 18 months, the child is expected to be using at least six words with appropriate meaning, and by 24 months, the child's speech is expected to be understood by familiar listeners at least 50% of the time.",
    "You don't need to worry about your child not being potty trained yet, as long as they are otherwise developing normally. It is normal for some children to take longer than others to master this skill, and there is no evidence to suggest that they will be at a higher risk of sexual abuse because of their age or lack of potty training.",
    "Children aged 6 should ideally have no more than 1–2 hours of screen time per day for recreational use.",
    "Encourage him to join activities that interest him, offer praise for small social steps, and be patient as he learns to socialize in his own way.",
    "Children with dyscalculia often struggle with basic math concepts. Consider consulting a learning specialist for a more accurate assessment.",
    "ADHD signs may include difficulty focusing, impulsivity, and hyperactivity. If you’re concerned, talk to your pediatrician for a professional evaluation.",
    "Many children begin reading by age 5 or 6, though development varies. Providing a literacy-rich environment helps foster early reading skills.",
    "Vaccines recommended for a 5-year-old include MMR, polio, and DTaP. Check with your healthcare provider to ensure your child’s immunizations are up-to-date.",
    "Yes, flu shots are generally safe for toddlers. It’s best to discuss any concerns with your pediatrician before vaccination.",
    "Yes, ibuprofen is safe for children when used at the correct dosage for their age and weight. Consult with a doctor for guidance.",
    "The MMR vaccine helps protect children from measles, mumps, and rubella, which can cause serious health issues if contracted.",
    "Offer comfort, explain why shots are important, and consider bringing a favorite toy for distraction.",
    "Calorie needs vary, but an average 10-year-old may require around 1,600 to 2,000 calories per day depending on activity level.",
    "Try offering a variety of healthy foods, allowing her to help with food prep, and consulting a pediatrician if you have concerns about nutrition.",
    "Healthy snacks include fruits, yogurt, cheese, and whole-grain crackers. Avoid foods high in sugar and salt.",
    "Multivitamins are generally unnecessary if your child has a balanced diet. Consult your pediatrician before starting supplements.",
    "Children should drink about 5–8 cups of water per day depending on their age, activity level, and climate.",
    "Avoid all peanut products, carry an epinephrine injector, and discuss an action plan with your pediatrician.",
    "Certain foods may trigger eczema. Consult an allergist to identify potential dietary triggers specific to your child.",
    "Signs of lactose intolerance include gas, diarrhea, and fussiness after consuming dairy. Consult your pediatrician for testing if symptoms persist.",
    "Common signs include hives, vomiting, trouble breathing, and swelling. Seek medical help immediately if your child has severe symptoms.",
    "Yes, some children outgrow peanut allergies, but many do not. Discuss your child’s allergy with an allergist for more information.",
    "Encourage open communication, offer reassurance, and consider establishing a routine to help him feel secure.",
    "Talk to your child and the school to address the issue, and consider building a support system to help your child cope.",
    "Teach her deep breathing techniques, encourage expression of feelings, and model calm behavior.",
    "Provide encouragement, recognize achievements, and help them set small, attainable goals.",
    "Yes, it’s normal and can be a healthy part of development for young children to have imaginary friends.",
    "A 6-year-old typically needs around 9–12 hours of sleep each night to support growth and development.",
    "Establish a calming bedtime routine, and try to respond calmly to night wakings to help them feel secure.",
    "Occasional snoring is normal, but if it’s frequent or loud, talk to a pediatrician as it could indicate an underlying issue.",
    "Be patient, encourage a bedtime routine, and limit liquids before bed. Bed-wetting is common and often resolves with time.",
    "A good bedtime routine includes a warm bath, storytime, and consistent sleep time to help them settle for the night.",
    "Monitor for signs of a concussion, like vomiting or confusion, and seek medical care if symptoms arise.",
    "Use outlet covers, secure furniture, and remove small objects to reduce risks.",
    "Rinse with cool water, avoid ice, and cover with a clean, non-stick bandage. Seek medical help for severe burns.",
    "Follow the pediatric CPR guidelines: 30 chest compressions and two breaths, then repeat. Consider taking a certified CPR course.",
    "Many children are ready to ride without training wheels around age 5–7, but ensure they wear a helmet and practice in a safe area.",
    "Average height varies, but many 5-year-olds are around 40–45 inches tall. Growth is individual, and regular check-ups can track progress.",
    "Offer nutrient-rich foods like whole grains, lean proteins, and healthy fats. Speak with a pediatrician for personalized guidance.",
    "Safe sports for young children include swimming, soccer, and running. Always ensure supervision and use of proper equipment.",
    "Encourage outdoor play, limit screen time, and join them in physical activities for fun family time.",
    "Light, supervised resistance exercises can be safe for children, but focus on body-weight exercises and consult a pediatrician for guidance."
]
tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train_sentences]

n = 2
train_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
words = [word for sent in tokenized_text for word in sent]
words.extend(["<s>", "</s>"])
padded_vocab = Vocabulary(words)
model = MLE(n)
model.fit(train_data, padded_vocab)

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else. Do not suggest anything if it's too dangerous to the user.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\n\nSources:" + str(sources)
    else:
        answer += "\n\nNo sources found"

    await cl.Message(content=answer).send()
    test_sentences = [res["result"]]
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]

    test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    # for test in test_data:
    #     print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

    test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    for i, test in enumerate(test_data):
        print("PP({0}):{1}".format(test_sentences[i], model.perplexity(test)))

