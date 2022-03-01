## Content Analysis

#you need to run the following to install the packages. Run each line one at a time. This should be run only once
#once the packages are installed everytime you open the code you DONOT need to run lines 5-13
install.packages("quanteda")
install.packages("corpus")
install.packages("topicmodels")
install.packages("LDAvis")
install.packages("tm")
install.packages("tmap")
install.packages("tidyverse")
install.packages("slam")
install.packages("tidytext")
install.packages("quanteda.textplots")
install.packages("quanteda.textstats")
install.packages("reshape2")
install.packages("dplyr")
install.packages("wordcloud")
install.packages("RColorBrewer")

#Select all lines 16 to 24 and then run the selection to use the packages
library(quanteda)
library(corpus)
library(topicmodels)
library(LDAvis)
library(tm)
library(tidyverse)
library(tidytext)
library(slam)
library(reshape2)
library(tmap)
library(tmaptools)
library(quanteda.textplots)
library(quanteda.textstats)
library(wordcloud)
library(RColorBrewer)
#load the data from csv file
#make sure the CSV file is formatted correctly, i.e. column names are correct.
#select lines 29-32 and then run the selection
data = read.csv("BDHE.csv") # please ensure the filename is correct and in the same folder as R script # nolint
View(data) # view that contents from file have been correctly loaded

data$Abstract <- as.character(data$Abstract)
data$Title <- as.character(data$Title)
data$DocNumber<- as.character(data$DocNumber)

# data preparation Part 1 - convert the data into a format R can analyse, which we call as corpus
#if you analyse a different text_field change Abstract, give the correct column name
#select the lines 39-44 and run the selection
corp <- corpus(data, text_field = 'Abstract')
corp <- corpus_reshape(corp, to = "paragraphs")

dfm <- dfm(corp)
dfm <- dfm_remove(dfm, remove_punct=T, remove=stopwords("english"), remove_numbers = TRUE)
dfm <- dfm_trim(dfm, min_docfreq = 5)
docid <- paste(data$DocNumber)
docnames(corp) <- docid

# data preparation part 2 - Creating a duplicate corpus to do some more analysis
# select the lines 48-55, and run the selection
import_corpus = Corpus(VectorSource(data$Abstract))
import_mat = 
  DocumentTermMatrix(import_corpus,control = list( #create root words
    stopwords = TRUE, #remove stop words
    minWordLength = 5, #cut out small words
    removeNumbers = TRUE, #take out the numbers
    removePunctuation = TRUE)) #take out punctuation 


################## VISUALISATION - Word Cloud##################
# wordcloud display showing most popular words
#change the numbers, i.e. maximum words to display depending on your needs
# run each line 61, 62, 63 and 67 - one at a time to see the outputs
textplot_wordcloud(dfm, max_words = 20)     ## top 20 (most frequent) words
textplot_wordcloud(dfm, max_words = 20, color = c('blue','red')) ## change colors
textstat_frequency(dfm, n = 100)         ## view the frequencies top 10 words

# create a word cloud for top 20 most frequent words in the middle and color=rainbow(7)
#rainbow is VIBGYOR, so Violet color word is most popular, followed by indigo, 
textplot_wordcloud(dfm, max_words = 100,random_order=FALSE, color=rainbow(7) ) 

# focus an analysis on whether and how documents talk about ai*
# if you want see on whether and how documents talk about innovation, use innovation*, 
#please note the * at the end of the word in line 70
# you should run each line one at a time to see the outputs
x <- as.corpus(x)
wordRel = kwic(tokens(corps ='higher education*'))
wordRel_corp = corpus(wordRel)
word_dtm = dfm(wordRel_corp, tolower=T, remove=stopwords('en'),  remove_punct=T)
textplot_wordcloud(word_dtm, max_words = 10)     ## top 10 (most frequent) words
textplot_wordcloud(word_dtm, max_words = 15,random_order=FALSE, color=rainbow(7) ) ## colored word cloud
textstat_frequency(word_dtm, n = 20)  

# I found two words using and usage, which are not relevant after executing line 76 above.
#i want to remove the words from the frequency table, so I will first run 81
#after executing line 81 then I will re-run lines 74, 75 and 76
word_dtm <- dfm_remove(word_dtm, c("using", "usage", "actual"), verbose= TRUE)
#after executing line 81 then I will re-run lines 74, 75 and 76 to see the new output/table

#if you want to see the document term matrix use for the whole dataset the following code 86 and 86 together
# document term matrix shows the frequency of each word in each abstract (document)
# for a large dataset it will take time, and your PC may hang, so leave it as is for few minutes
word_dtm1 = dfm_remove(corp, tolower=T, remove=stopwords('en'), remove_punct=T)
View(word_dtm1)

# after executing lines 87 and 88, you may see many zeroes, we want to remove rows having many zeroes because
# it will be good for analysis
# select lines 93-100 and run them together
import_weight = tapply(import_mat$v/row_sums(import_mat)[import_mat$i], 
                       import_mat$j, 
                       mean) *
  log2(nDocs(import_mat)/col_sums(import_mat > 0))

#ignore very frequent and 0 terms
import_mat = import_mat[ , import_weight >= .1]
import_mat = import_mat[ row_sums(import_mat) > 0, ]

# the following creates topics, i.e. cluster words together
#note that you need to give meaningful names to topics once you get them
# topics are words clustered together
# select lines 106-115 and run it together. 
dtm = convert(dfm, to = "topicmodels") 
#set.seed(1234)
#you can change the value of k, i.e. number of topics you want the program to find
m = LDA(dtm, method = "Gibbs", k = 7,  control = list(seed = 1234, burnin = 1000, 
                                                      thin = 100, iter = 1000))

LDA_gibbs = LDA(import_mat, k = 7, method = "Gibbs", 
                control = list(seed = 1234, burnin = 1000, 
                              thin = 100, iter = 1000))

# now we calculate alpha values
# higher value of aplha is good for GIBBS algorithm
# change the value of k to 3 in lines 109, 110, 114 to see how alpha value changes, if it lowers thats good
# select lines 121-123 and then run the selection
#m@alpha
LDA_gibbs@alpha
#LDA_fit@alpha
# now we calculate entropy 
# higher value is good meaning topic is evenly spread across documents
# select lines 128 to 130 and then run the selection
sapply(list(m, LDA_gibbs), 
       function (x) 
         mean(apply(posterior(x)$topics, 1, function(z) - sum(z * log(z)))))


#following code shows five topics, and 5 words for each topic, you can
#change the 5 to a different value if you want to see more words fo a topic
# select lines 137-140 and then run the selection
# you will see outputs of 3 algorithms, use the one which you feel is the best
terms(m, 10)
terms(LDA_gibbs,10)

# see top 5 words in a topic with their weighing, i.e. how important they are in that topic
#lower score means less important
#select lines 144-147 and run it togetherWords i
topic = 5
words = posterior(LDA_gibbs)$terms[topic, ]
topwords = head(sort(words, decreasing = T), n=20)
head(topwords)


################ VISUALISATION - TOPICS AND WORDS#####################
# select lines153-174 and run it together
#create a top terms 
LDA_gibbs_topics = tidy(LDA_gibbs, matrix = "beta")

top_terms = LDA_gibbs_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

cleanup = theme(panel.grid.major = element_blank(), 
                panel.grid.minor = element_blank(), 
                panel.background = element_blank(), 
                axis.line.x = element_line(color = "black"),
                axis.line.y = element_line(color = "black"),
                legend.key = element_rect(fill = "white"),
                text = element_text(size = 10))
top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  cleanup +
  coord_flip()

# now if you want to see the importance of all terms in each topic
# select lines 178 and 179, run it together
distr = dcast(LDA_gibbs_topics, term~topic, value.var ="beta")
View(distr)

#heatmap showing the relationship between documents and topics
# x axis is topics
#y axis is documents (abstract from papers), i.e. each row in your CSV file
# select lines 185-190 and run it together
docs = docvars(dfm)[match(rownames(dtm), docnames(dfm)),]
#please make sure the first column in your file is named DocNumber and 
#it is a string - comprises of charcters
tpp = aggregate(posterior(m)$topics, by=docs["DocNumber"], mean)
rownames(tpp) = tpp$DocNumber
heatmap(as.matrix(tpp[-1]))

# now we want to view the importance of topics in each document in a table (higher values means good fit, i.e. close to 1)
# this will help to select which topic best represents a document in a table
# select and run lines 194-197 together 
m_topics = tidy(m, matrix = "gamma")
distr1 = dcast(m_topics, document~topic, value.var ="gamma")
View(distr1)
# another way to visualise the distribution of topics in documents
#select and run lines 200-204
LDA_gamma=m_topics
LDA_gamma %>%
  ggplot(aes(factor(topic), gamma)) +
  geom_point() 

# Representation of topics over documents
# we have five topics
# lower values < 0.15 means that topic can be merged to another topic, so reduce the number of topics
# select lines 210-214 and run them together
sum(distr1$'1')/nrow(distr1)
sum(distr1$'2')/nrow(distr1)
sum(distr1$'3')/nrow(distr1)
sum(distr1$'4')/nrow(distr1)
sum(distr1$'5')/nrow(distr1)


#create the cognitive map to see the relationships between topics
#this will help you to find themes and you have to name themes meaningfully
#your output will open in a browser in a new tab [tested in Google Chrome]
# select lines 220-228 and run them together
dtm = dtm[slam::row_sums(dtm) > 0, ]
phi = as.matrix(posterior(m)$terms)
theta <- as.matrix(posterior(m)$topics)
vocab <- colnames(phi)
doc.length = slam::row_sums(dtm)
term.freq = slam::col_sums(dtm)[match(vocab, colnames(dtm))]
json = createJSON(phi = phi, theta = theta, vocab = vocab,
                  doc.length = doc.length, term.frequency = term.freq)
serVis(json)

