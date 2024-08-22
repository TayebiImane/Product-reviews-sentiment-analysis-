import pandas as pd
from textblob import TextBlob


D = pd.read_excel('C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Tayebi_Imane_Amazon_Product_Reviews_Collected_Data.xlsx')#the path should be Replaced with actual file path
print(D)
i=0
def Text_correction(text):
    global i 
    if pd.isna(text):
        return text
    blob = TextBlob(text)
    print(str(blob.correct()))
    i+=1
    print(i)
    return str(blob.correct())
D["Reviews.text"]=D["Reviews.text"].apply(Text_correction)
D.to_excel("C:/Users/Utilisateur/Desktop/Product_review_sentimeny_Analysis/Tayebi_Imane_Amazon_Product_Reviews_Corrected_Text_Data.xlsx")
#Execution time : 2h 30 min
