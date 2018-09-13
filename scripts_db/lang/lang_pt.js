db.tweets.find({'lang':'pt'}).forEach(function(obj) {
    print(obj.full_text.valueOf().toString());
});

