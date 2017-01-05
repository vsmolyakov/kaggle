import re
import csv
import operator
from collections import defaultdict
from nltk.corpus import stopwords

def f1_score(tp, fp, fn):    
    p = (tp*1.0)/(tp+fp)
    r = (tp*1.0)/(tp+fn)
    f1 = (2.0*p*r)/(p+r)    
    return f1
    
def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext
    
def get_words(text):
    word_split = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    return [word.strip().lower() for word in word_split.split(text)]
    

if __name__ == "__main__":
    
    in_file = open('./data/test.csv')
    out_file = open('./word_freq.csv', 'w')

    reader = csv.DictReader(in_file)
    writer = csv.writer(out_file)
    writer.writerow(['id','tags'])
    
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    
    for ind, row in enumerate(reader):
        text = clean_html(row['title'])
        tfrequency_dict = defaultdict(int)

        word_count = 0
        for word in get_words(text):
            if word not in stop_words and word.isalpha():
                tfrequency_dict[word] += 1
                word_count += 1

        #normalize by word count
        for word in tfrequency_dict:
            tf = tfrequency_dict[word] / float(word_count)
            tfrequency_dict[word] = tf
        
        pred_title_tags = sorted(tfrequency_dict, key = tfrequency_dict.get, reverse=True)[:10]
        
        text = clean_html(row['content'])
        dfrequency_dict = defaultdict(int)
    
        word_count = 0
        for word in get_words(text):
            if word not in stop_words and word.isalpha():
                dfrequency_dict[word] += 1
                word_count += 1

        #normalize by word count
        for word in dfrequency_dict:
            tf = dfrequency_dict[word] / float(word_count)
            dfrequency_dict[word] = tf
    
        pred_content_tags = sorted(dfrequency_dict, key = dfrequency_dict.get, reverse=True)[:10]
        
        pred_tags_dict = {}
        for word in set(pred_title_tags + pred_content_tags):
            pred_tags_dict[word] = tfrequency_dict.get(word,0) + dfrequency_dict.get(word,0)
        pred_tags = set(sorted(pred_tags_dict, key = pred_tags_dict.get, reverse=True)[:3])
        
        writer.writerow([row['id'], " ".join(pred_tags)])
        
        if ind % 10000 == 0:
            print "Processed: ", ind
        
    in_file.close()
    out_file.close()    
                
                                
    