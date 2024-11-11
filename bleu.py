from nltk.translate.bleu_score import sentence_bleu

reference = ["Signs of dehydration in kids include dry mouth and lips, fewer wet diapers or trips to the bathroom, sunken eyes or soft spot in infants, no tears when crying, dark urine, fatigue or irritability, and dry, cool skin. Encourage fluids and contact a healthcare provider if symptoms persist or seem severe.",
             "If you suspect that your child may be dehydrated, you should check for signs of dehydration such as dry mouth, dark yellow or brown urine, fewer wet diapers during the day, lethargy, loss of appetite, decreased urine output when drinking fluids, and irritability."]

reference = [ref.split() for ref in reference]

candidate = "If you suspect that your child may be dehydrated, you should check for signs of dehydration such as dry mouth, dark yellow or brown urine, fewer wet diapers during the day, lethargy, loss of appetite, decreased urine output when drinking fluids, and irritability.".split()


print('BLEU score -> {}'.format(sentence_bleu(reference, candidate)))
