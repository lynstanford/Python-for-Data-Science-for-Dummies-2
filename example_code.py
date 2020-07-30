def comparison_test(text):
    import sklearn.feature_extraction.text as txt
    htrick = txt.HashingVectorizer(n_features=20, 
                                   binary=True, 
                                   norm=None) 
    oh_enconder = txt.CountVectorizer()
    oh_enconded = oh_enconder.fit_transform(text)
    hashing = htrick.transform(text)
    return oh_enconded, hashing
