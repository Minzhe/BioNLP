# Run standford core nlp program to extract eneities.

cd ~/software/stanford-corenlp-full-2018-02-27
train_path=~/project/BioNLP/BioNLP-ST-2016_SeeDev/data/BioNLP-ST-2016_SeeDev-binary_train

# train
for f in `ls $train_path/*.txt`
do
    java -cp "*" -Xmx8g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,parse,ner,dcoref -file $f -outputDirectory $train_path -outputFormat "text"
done

# test
test_path=~/project/BioNLP/BioNLP-ST-2016_SeeDev/data/BioNLP-ST-2016_SeeDev-binary_test
for f in `ls $test_path/*.txt`
do
    java -cp "*" -Xmx8g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,parse,ner,dcoref -file $f -outputDirectory $test_path -outputFormat "text"
done

# dev
dev_path=~/project/BioNLP/BioNLP-ST-2016_SeeDev/data/BioNLP-ST-2016_SeeDev-binary_dev
for f in `ls $dev_path/*.txt`
do
    java -cp "*" -Xmx8g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,parse,ner,dcoref -file $f -outputDirectory $dev_path -outputFormat "text"
done