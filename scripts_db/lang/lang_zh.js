db.tweets.find({'lang':'zh'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

