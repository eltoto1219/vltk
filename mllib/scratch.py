# Build Class
    # conf = LxmertConfig(num_qa_labels=1536)
    # model = LxmertForQuestionAnswering(conf)
    # model = model.from_pretrained("unc-nlp/lxmert-base-uncased", cache_dir="ssd-playpen/avmendoz")
    # gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased", cache_dir="ssd-playpen/avmendoz")
    gqa = GQA()
    # Test or Train
    # if args.test is not None:
    if True:
        print("here")
        args.fast = args.tiny = False  # Always loading all data in test
        # if "submit" in args.test:
        #     raise Exception("not testing")
        #     gqa.predict(
        #         get_tuple(
        #             args.test, bs=args.batch_size, shuffle=False, drop_last=False
        #         ),
        #         dump=os.path.join(args.output, "submit_predict.json"),
        #     )
        if True:
            result = gqa.evaluate(
                get_tuple("valid", bs=32, shuffle=False, drop_last=False),
                dump=os.path.join(args.output, "testdev_predict_aug.json"),
            )
            print(result)
    else:
        # raise Exception("model is not testing")
        print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print("Splits in Train data:", gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print("whao")
            print("Splits in Valid data:", gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)
