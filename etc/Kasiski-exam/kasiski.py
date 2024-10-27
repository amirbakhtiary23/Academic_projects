import numpy as np
import pandas as pd
def string_based_attack(step,first_char,text):
    messages=[]
    for i in range(step):
        messages.append("")
        for j in range(i,len(text),step):
            messages[i]=messages[i]+text[j]
    return messages

def analysis(first_freq,matrix):
    maximum=0;
    max2=0;
    index=0;
    index_2=0;
    for i in range(len(matrix)):
        sum_=np.dot(first_freq,matrix[i])
        if sum_>max2:
            if sum_>maximum:
                maximum=sum_
                index=i
            else :
                max2=sum_
                index_2=i
    return index,index_2,maximum,max2


def freq(text):
    length=len(text)
    freq_letters=[]
    for i in alphabet:
        counter=0
        for j in text:
            if i==j:
                counter+=1
        freq_letters.append(round((counter/length),3))
    return freq_letters

def vigenere(
        text: str, 
        key: str, 
        alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        encrypt=True
):

    result = ''

    for i in range(len(text)):
        letter_n = alphabet.index(text[i])
        key_n = alphabet.index(key[i % len(key)])

        if encrypt:
            value = (letter_n + key_n) % len(alphabet)
        else:
            value = (letter_n - key_n) % len(alphabet)

        result += alphabet[value]

    return result

def vigenere_encrypt(text: str, key: str):
    return vigenere(text=text, key=key,alphabet=alphabet, encrypt=True)


def vigenere_decrypt(text: str, key: str):
    return vigenere(text=text, key=key,alphabet=alphabet, encrypt=False)
    
#brute_force substring finder
def sub(text):#the minimum is 3
    substrings={}
    length=len(text)
    maximum=length//2
    for i in range(3,maximum):
        for j in range(0,length):
            word=text[j:j+i]

            step=0;
            while True:
                
                if text[j+step+i:j+i+i+step]=="":
                    break
          
                if word==text[j+step+i:j+i+i+step]:
                    try :
                        
                        substrings[word].append(step+i)
                                                             
                    except KeyError:
                        
                        substrings[word]=[step+i]
                step+=1
    return substrings

    
#pre_processing
text= """LFWKI MJCLP SISWK HJOGL KMVGU RAGKM KMXMA MJCVX WUYLG GIISW
ALXAE YCXMF KMKBQ BDCLA EFLFW KIMJC GUZUG SKECZ GBWYM OACFV
MQKYF WXTWM LAIDO YQBWF GKSDI ULQGV SYHJA VEFWB LAEFL FWKIM
JCFHS NNGGN WPWDA VMQFA AXWFZ CXBVE LKWML AVGKY EDEMJ XHUXD
AVYXL
"""
alphabet=sorted(list("qwertyuiopasdfghjklzxcvbnm".upper()))
special_chars =["!",",",".","?",'"',"'","-","(",")",","]
spaces = []
parentheses=[]
for i in enumerate (text):
    if i[1]==" " or i[1]=="\n":spaces.append(i[0])

text=text.split(" ")
text = "".join(text)
text="".join(text.split("\n"))
text=list(text)

for i in enumerate (text):
    if i[1] in special_chars:
        parentheses.append(i)
        
for i in reversed(range(len(parentheses))):
    text.pop(parentheses[i][0])
    
text = "".join(text).upper()


substrings=sub(text)
print ("done")

cds={}
for i in substrings.values(): 
    #print (i)
    for j in range (3,20):
        if min(i)%j==0:
            #print (i[0],j)
            try:
                cds[j]=cds[j]+1
            except:
                cds[j]=1
print (cds)
cds=list(reversed(sorted(cds.items(), key=lambda item: item[1])))
print(cds)
first_char=text[0]

attacks=[]
for i in range(4):#choosing the first m possiblity, 4 is m in this case
    attacks.append((cds[i][0],string_based_attack(cds[i][0],first_char,text)))
frequency_table=pd.read_csv("english-letter.csv")
frequency_vector=[]
for i in np.array(frequency_table):
    frequency_vector.append(i[1])
frequency_vector=np.array(frequency_vector)
frequency_matrix=[]
for i in range(0,26):
    frequency_matrix.append(np.roll(frequency_vector,i))

frequency_matrix=np.array(frequency_matrix)
#finding frequency of each letter in ciphertext

each_char_frequency=[]
for j in range(len(attacks)):
    each_char_frequency.append([])
for i in range(len(attacks)):
    for j in range(attacks[i][0]):
        each_char_frequency[i].append(np.array(freq(attacks[i][1][j])))

each_char_frequency=np.array(each_char_frequency,dtype=object)

keys=[]
for i in each_char_frequency:
    key=""
    for j in range(len(i)):
        key=key+alphabet[analysis(i[j],frequency_matrix)[0]]
        #print (analysis(i[j],frequency_matrix))
    keys.append(key)
    print ("possible keys for length ",len(key)," is :",key)



for i in keys:
    txt=vigenere_decrypt(text, i)
    
    print ("\n\n\nfor keyword ",i,txt)
txt=vigenere_decrypt(text, "DRSMOHAMMADI")
print ("\n\n\nfor keyword DRSMOHAMMADI",txt)

