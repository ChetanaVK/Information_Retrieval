#!/usr/bin/env python
# coding: utf-8

#Code For the Search engine


get_ipython().system('pip install contractions')

env='/kaggle/input'

import os
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import contractions
lemmatizer = WordNetLemmatizer() 
ps = PorterStemmer()
def preprocess():
    for dirname,_, filenames in os.walk('/kaggle/input/televisionsss'):
        #filenames=filenames[2:]
        #print(filenames)

        d=dict()
        d_num=1

        final_list=[]#this has 417 list of lists
        orig=[]
        stop_words = set(stopwords.words('english')) #Contains all the stop words
        for filename in filenames:
            #print(filename)
            d[d_num]=filename
            #print(d_num)
            col_list = ['Snippet']
            
            #print(os.path.join(dirname,filename))
            content=pd.read_csv(os.path.join(dirname,filename),sep=',',usecols=col_list)
            #print(content)
            doc=content.to_string().split('\n')
            doc=doc[1:]
            #print(doc[0])
            inter_list=[]#this had tokens for each row of ONE doc
            inter_original=[]
            for row in doc:
                cache=[]
                #print(type(row))
                #print(row)
                original_tokens=[]
                row=row.lower()
                #print(row)
                #row = row.replace("'","")                             
                no_num=re.sub(r"[^a-zA-Z0-9]+", ' ', row[1:])
                #print(no_num)
                no_num = contractions.fix(no_num)
                #print(no_num)
                tokens=word_tokenize(no_num)
                #print(tokens)
                
                
                #print(orig)
                for word in tokens:
                    if len(word)>1:
                        original_tokens.append(word)
                        if word not in stop_words:
                            cache.append(ps.stem(word))

                    #print(cache)
                #print(cache)
            
                #res = list(OrderedDict.fromkeys(cache)) 
                inter_list.append(cache)
                inter_original.append(original_tokens)
            orig.append(inter_original)
            #print(original_tokens)
            #print(inter_list)
            final_list.append(inter_list)       
            d_num+=1
        #print(final_list[0])
        #print(d)
        return d,final_list,orig


'''if __name__ == "__main__":
    di=dict()
    li=list()
    original_toks=[]
    di,li,original_toks=preprocess()
    print(original_toks[0])'''


#creates indiex for each file
import math
from heapq import *

def create_index_vector_space_model(tokens):
    
    index=dict()
    length_of_row=[]
    c=1
    for row_tokens in tokens:
        row_term=dict()
        l=0
        for row_token in row_tokens :
            if row_token not in row_term:
                row_term[row_token]=1
            else:
                row_term[row_token]+=1
            l+=1
        n=0
        #row_term ={"british":3,"me":4} [{}]
        for i in row_term.keys():
            freq=row_term[i]
            freq=1+math.log(freq)
            n+=freq*freq
        #btree here 
        n=n**0.5
        for i in row_term.keys():
            freq=row_term[i]
            if(i in index):
                index[i][0]+=1
                #index[i][1].append([c,round((1+math.log(freq))/n,5)])
                index[i][1].append([c,round((1+math.log(freq)),5)])
            else:
                index[i]=[1,list()]
                #index[i][1].append([c,round((1+math.log(freq))/n,5)])
                index[i][1].append([c,round((1+math.log(freq)),5)])
        c=c+1
        length_of_row.append(l)
    return [index,length_of_row]

'''indexes=[]
#creates index for all files
for i in li:
    indexes.append(create_index_vector_space_model(i))'''
    

#positional index

def create_positional_index(l_li):
    list_of_position_index=[]
    for doc in l_li:
        position_index=dict()
        r=0
        for row in doc:
            p=0
            temp=dict()
            for token in row:
                if token not in temp:
                    temp[token]=[p]
                else:
                    temp[token].append(p)
                p=p+1
            #print(temp)
            for key in temp.keys():
                if(key not in position_index):
                    position_index[key]=[(r,temp[key])]
                else:
                    position_index[key].append((r,temp[key]))
            r=r+1
        list_of_position_index.append(position_index)
        
    return list_of_position_index     
            


#create_positional_index(li)    
                


#getting topk from original index
hque=[]
def topk(docid,row,score):
    l=len(hque)
    if l<30:
        flag=True
        val=str(docid)+'#'+str(row)
        for i in hque:
            if(i[1]==val):
                flag=False
                break
        if(flag==True):
            heappush(hque, (score,val ))
    else:
        smallest=hque[0][0]
        
        if(score>smallest):
            heappop(hque)
            val=str(docid)+'#'+str(row)
            heappush(hque, (score,val))
    #print(hque)


def compute_score(query,index,number_of_rows,docid):
    score=dict()
    #160-ours
    #244-es
    result=[]
    length=index[1]
    index=index[0]
    query_freq=dict()
    for i in query:
        if i in query_freq:
            query_freq[i]+=1
        else:
            query_freq[i]=1
    cw=0
    for i in query_freq.keys():
        query_freq[i]=1+math.log(query_freq[i])
        n=query_freq[i]
        n=n*n
        cw+=n
    cw=cw**0.5
        
    for i in query:
        if(i in index ):
            #print("posting list of "+i)
            posting_list=index[i][1]
            idf=math.log(number_of_rows/index[i][0])
            #print(posting_list)
           
            #wtq=(query_freq[i])*idf/cw
            wtq=(query_freq[i])*idf
            for row_number_wtd in posting_list:
                if row_number_wtd[0] in score:
                    score[row_number_wtd[0]]+=row_number_wtd[1]*wtq
                else:
                    score[row_number_wtd[0]]=row_number_wtd[1]*wtq
    #print(score)
    for s in score.keys():
        #score[s]=round(score[s]/length[s-1],5)
        #if(docid==160 or docid== 244):
        #   print(docid,s ,round(score[s]/length[s-1],5))
        #topk(docid,s,round(score[s]/length[s-1],5))
        topk(docid,s,round(score[s],5))
        #print(type(docid))
        try:    
            result.append(di[docid]+"-"+str(s))
        except:
            print(docid,s)
        #print("SADF")
    #print(score)
    #sorted_scores=[k for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)][:10]
    #print(yo)
    #print(score)
    #return sorted_scores
    return result
        
    
#function to normalise query terms
def preprocess_query(query):
    terms=[]
    stop_words = set(stopwords.words('english'))
    no_num=re.sub(r"[^a-zA-Z0-9]+", ' ',query )
    #print(no_num)
    no_num = contractions.fix(no_num)
    #print(no_num)
    tokens=word_tokenize(no_num)
    #print(tokens)
    for word in tokens:
        if word not in stop_words and len(word)>1:
            terms.append(ps.stem(word))
    return terms

def eliminate_doc(terms):
    #no_terms=int(0.95*len(terms))#3,7
    no_terms=0
    miss_terms=len(terms)-no_terms#4
    chklist=[miss_terms]*417
    for t in terms:
        for d in range(417):
            if t not in indexes[d][0]:
                chklist[d]-=1
    yeslist=[]#has docs that have 75% or more query terms
    for i in range(417):
        if(chklist[i]>=0):
            yeslist.append(i)
    #print(yeslist)
    return yeslist  
    
    
    
#use the heapq or priority queue for sorting.          
#compute cosine for all indexes

def process(query):
    query=preprocess_query(query)
    #print(query)
    results=[]
    #scores=dict()
    yeslist=eliminate_doc(query)
    #print(yeslist)
    for i in yeslist:
        results=results+compute_score(query,indexes[i],len(li[i]),i+1)
    #print(hque)  
    #return scores
    #print(len(results))
    #print(results)
    
def display():
    finalk=[]
    #print(hque)
    ln=len(hque)-1
    if(ln==-1):
        print("Sorry! No results found:(")
    
    while(hque!=[]):
        
        #finalk.append(heappop(hque)[1].split('#'))
        inter=heappop(hque)[1].split('#')
        docid=int(inter[0])
        docname=di[docid]
        finalk.append(docname+'-'+inter[1])
    for i in range(ln,-1,-1):
        print(finalk[i])
    #print(finalk)
        


#calculations for positional index 
def intersection_positional_index(query):
    list_of_terms=preprocess_query(query)
    #print(list_of_terms)
    listt=dict()
    flag=True
    docid=1
    if(len(list_of_terms)==1):
        #print("here")
        process(query)
        
    elif(len(list_of_terms)>=2):
       
        for index in list_of_position_index:
            flag=True
            for terms in list_of_terms:
                if(terms not in index):
                    flag=False
            concat=[]
            #print("flag",flag)
            if(flag==True):
                if(len(list_of_terms)>=2):
                    concat=intersection_positional_index_two_terms(index[list_of_terms[0]],index[list_of_terms[1]])
                    for i in range(2,len(list_of_terms)):
                            concat=intersection_positional_index_two_terms(concat,index[list_of_terms[i]])
                    #print("inside true block",concat)
                else:
                    concat=index[list_of_terms[0]]
            li=[]
            for r,list_of_positions in concat:
                if(len(list_of_positions)>0):
                    li.append((r+1,len(list_of_positions)))#row number,phase frequency
            if(concat!= []):
                listt[docid]=li
            docid=docid+1  
        #print(hque)
        for d in listt.keys():
            for row_score in listt[d]: 
                topk(d,row_score[0],100+row_score[1])
        #print(hque)
        #return listt
        #print(listt)
    
        
                
                

def  intersection_positional_index_two_terms(pl1,pl2):
    p1=len(pl1)
    p2=len(pl2)
    c1=0
    c2=0
    final_rows=[]
    while(c1<p1 and c2<p2):
        if(pl1[c1][0]==pl2[c2][0]):
            a=pl1[c1][1]
            b=pl2[c2][1]
            a1=0
            b1=0
            c=[]
            for i in a:
                if(i+1 in b):
                    c.append(i+1)
            final_rows.append((pl1[c1][0],c))#row number ,positions of the secod term
            c1=c1+1
            c2=c2+1
        elif(c1<p1 and pl1[c1][0]<pl2[c2][0]):
            while(c1<p1 and pl1[c1][0]<pl2[c2][0]):
                c1=c1+1
        elif(c2<p2 and pl2[c2][0]<pl1[c1][0]):
            while(c2<p2 and pl2[c2][0]<pl1[c1][0]):
                c2=c2+1
    return final_rows
            
                
    

#list of list of list

def create_trigrams_index(corpus):
    d=1
    for doc in corpus:
        for row in doc:
            #print(row)
            for term in row:
                #print(term)
                ter = '$'+term+'$'
                for i in range(0, len(ter)-2):
                    tri_gram=ter[i:i+3]
                    #print(tri_gram)
                    if tri_gram in kgram:
                        #{"anv":[("anvitha",[23,56]),("anvitihing",[54,7,8])]}
                        posting_list=kgram[tri_gram]
                        lenn=len(posting_list)
                        for j in range(lenn):
                            if(term<posting_list[j][0]):
                                kgram[tri_gram].insert(j,(term,[d]))
                                break
                            elif(term==posting_list[j][0]):
                                if(d!=kgram[tri_gram][j][1][-1]):
                                    kgram[tri_gram][j][1].append(d)
                                break
                        if(j==lenn-1 and term!=kgram[tri_gram][j][0]):
                            kgram[tri_gram].append((term,[d]))
                    else:
                        kgram[tri_gram]=[(term,[d])]
        #print(d)
                    
        d=d+1
#create_trigrams_index(inputt)
#print(kgram)



def intersect_tri_grams(f_list_tri):
    intersection_list=[]
    #print("here")
    #print(list_trigrams)
    list_trigrams=[]
    for term in f_list_tri:
        if term in kgram:
            list_trigrams.append(term)
    lenn=len(list_trigrams)
    #print(list_trigrams)
    #print("start",lenn)
    if(lenn>=2):
        intersection_list=inserction_2_tri_grams(kgram[list_trigrams[0]],kgram[list_trigrams[1]])
        #print(intersection_list)
        for i in range(2, lenn):
            intersection_list=inserction_2_tri_grams(intersection_list, kgram[list_trigrams[i]])
            #print(intersection_list)
    
    elif(lenn==1):
        intersection_list=kgram[list_trigrams[0]]
    #print("Done",intersection_list)
    return intersection_list
            
        
                       
def inserction_2_tri_grams(pl1,pl2):
    #two_merge_flist=[value for value in pl1 if value in pl2]
    p1=0
    p2=0
    intersect=list()
    while(p1<len(pl1) and p2<len(pl2)):
        if(pl1[p1][0]==pl2[p2][0]):
            intersect.append((pl1[p1][0],list(set(pl1[p1][1]+pl2[p2][1]))))
            pl=p1+1
            p2=p2+1
        elif(pl1[p1][0]<pl2[p2][0]):
            while(p1<len(pl1) and pl1[p1][0]<pl2[p2][0] ):
                p1+=1
        elif(pl2[p2][0]<pl1[p1][0]):
            while(  p2<len(pl2) and pl2[p2][0]<pl1[p1][0]):
                p2+=1
    #print(intersect)
    return intersect
    
#print(intersect_tri_grams([('anvitha',[1,2]),('mahimas',[1,2,7])],[('anvitha',[1,2,7]),('chetanas',[3,6,7]),('mahima',[3,6,3])],
#                            [('anvitha',[1,2,9,10]),('chetanas',[3,6,7]),('mahima',[3,6,3])]))
            

import re   
def string_matching(query,common_words_across_trigrams):
    final_words=[]
    set_of_documents=set()
    #print("q",query)
    #print("common",common_words_across_trigrams)
    query=query.replace('*', '.*')
    query=query.replace('?', '.')
    #print("qq",query)
    for word in common_words_across_trigrams :
        x = re.search(query, word[0])
        if(x):
            final_words.append(word[0])
            set_of_documents.update(word[1])
    #print(final_words,set_of_documents)
    return (final_words,list(set_of_documents))
        
    
    
def generate_trigrams(word):
    tri_grams=[]
    for i in range(0, len(word)-2):
                    tri_gram=word[i:i+3]
                    tri_grams.append(tri_gram)
    return tri_grams

    

def wcq_preprocess(word):
    stop_words = set(stopwords.words('english'))
    
    lemmatizer = WordNetLemmatizer() 
    ps = PorterStemmer()
    word=word.lower()
                                      
    no_num=re.sub(r"[^a-zA-Z0-9]+", ' ', word)
    #print(no_num)
    no_num = contractions.fix(no_num)
    #print(no_num)
    tokens=word_tokenize(no_num)
    if(word in stop_words):
        return ''
    fin_word=ps.stem(word)
    return fin_word


def process_wildcard_query(query):
    tri_grams=[]
    q=query
    ls=False
    rs=False
    if(len(query)>1):
        if(query[0]=='*' or query[0]=='?'):
            ls=True
    value=""
    tri_grams_terms=[]
    for ch in query:
        if(ch != '*' and ch!='?'):
            value=value+ch
        else:
            rs=True
            if(ls==True and rs==True):
                ls=True
                tri_grams_terms.append(value)
                value=""
            elif(rs==True and ls==False):
                value='$'+value
                tri_grams_terms.append(value)
                value=''
                ls=True
    if(value != ""):
        value=value+'$'
        tri_grams_terms.append(value)
    for i in tri_grams_terms:
        tri_grams.extend(generate_trigrams(i))
    tri_grams=list(set(tri_grams))
    #print(tri_grams_terms)
    #print(tri_grams)
        
    common_words_across_trigrams=intersect_tri_grams(tri_grams)
    #print("common",common_words_yes

    final_words,set_of_documents=string_matching(q,common_words_across_trigrams)
    #print(final_words,set_of_documents)
    final_lematized_words=[]
    for words in final_words:
        final_lematized_words.append(wcq_preprocess(words))
    #print(final_lematized_words)
    for docid in set_of_documents:
        #print(docid)
        #print(final_lematized_words[0] in indexes[docid-1][0])
        compute_score(final_lematized_words,indexes[docid-1],len(li[docid-1]),docid)
    
    #return (final_words,set_of_documents)
        

    
#print(intersect_tri_grams(['che','het','eta'])) 
                       
                    
                
                


'''#saving the data     
import pickle
def save_data():
    with open('kgram.pickle','wb') as f:
        pickle.dump(kgram,f,pickle.HIGHEST_PROTOCOL)
    with open('di.pickle','wb') as f:
        pickle.dump(di,f,pickle.HIGHEST_PROTOCOL)
    with open('li.pickle','wb') as f:
        pickle.dump(li,f,pickle.HIGHEST_PROTOCOL)
    with open('original_toks.pickle','wb') as f:
        pickle.dump(original_toks,f,pickle.HIGHEST_PROTOCOL)'''

import pickle
kgram=dict()
indexes=[]
#create_trigrams_index(original_toks)    
di=dict()
li=[]
original_tokens=[]

di_to_read = open(env+"/di.pickle", "rb")

kgram_to_read = open(env+"/kgram.pickle", "rb")
li_to_read = open(env+"/li.pickle", "rb")
orig_to_read = open(env+"/original_toks.pickle", "rb")
kgram= pickle.load(kgram_to_read)
di=pickle.load(di_to_read)
li=pickle.load(li_to_read)
original_tokens=pickle.load(orig_to_read)
for i in li:
    indexes.append(create_index_vector_space_model(i))
list_of_position_index=create_positional_index(li) 
  
    


#UI
import time
import copy
def lets_query(query,choice):#choice:1=phrase query,2=wildcard,3=normal search
    if(choice==1):
        start = time.time()
        global hque
        #print(hque)
        #hque=[]
        intersection_positional_index(query)
        #print(hque)
        
        #print(type(hque))
        if(len(hque)<10):
            #print("lesser",(hque))
            process(query)
            #print(hque)
        to_ret_hq=copy.deepcopy(hque)    
        display()
        end = time.time()
        print("Time taken to calculate",end - start, "in seconds")
        return to_ret_hq
    elif(choice==2):
        start = time.time()
        hque=[]
        process_wildcard_query(query)
        to_ret_hq=copy.deepcopy(hque)   
        display()
        end = time.time()
        print("Time taken to calculate",end - start, "in seconds")
        return to_ret_hq
    elif(choice==3):
        start = time.time()
        hque=[]
        process(query)
        to_ret_hq=copy.deepcopy(hque)   
        display()
        end = time.time()
        print("Time taken to calculate",end - start, "in seconds")
        return to_ret_hq
    #print(hque)
    return hque



# %% [code]
f=open(env+"/test.txt")
for line in f:
    #print(line)
    ch=line.split(",")[0]
    ch=int(ch)
    q=line.split(",")[1][:-1]
    print()
    print(q)
    #lets_query(q,ch)
    lets_query(q,ch)


