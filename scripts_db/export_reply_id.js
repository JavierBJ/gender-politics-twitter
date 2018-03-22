db.tweets.find({"msg":"reply"}).forEach(function(obj) {
    print(obj.in_reply_to_status_id);
});
