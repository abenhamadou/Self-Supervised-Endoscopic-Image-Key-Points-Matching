
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FtqIVg3rcY4cnBTvxBGFxhXOgiX7ZI27' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FtqIVg3rcY4cnBTvxBGFxhXOgiX7ZI27" -O ../data/test_dataset.zip && rm -rf /tmp/cookies.txt



wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12adtTzaVbZ_giFJLnmd4K-rq-o9iOWZ3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12adtTzaVbZ_giFJLnmd4K-rq-o9iOWZ3" -O ../data/train_dataset.zip && rm -rf /tmp/cookies.txt


cd ../data
rm -fr test_dataset
rm -fr base_train2
unzip test_dataset.zip 
unzip train_dataset.zip
rm test_dataset.zip 
rm train_dataset.zip


#wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=12adtTzaVbZ_giFJLnmd4K-rq-o9iOWZ3' -O ../data/train_dataset.zip
