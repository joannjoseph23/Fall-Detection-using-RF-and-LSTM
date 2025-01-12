


from featuresExtraction_WEDA import extract_features_WEDA


def main():
    
    
    t_window = ('3&1.5',)
   
    pasta_WEDA = '/Users/nityareddy/Desktop/AIML PROJECT/normalized'
    extract_features_WEDA(pasta_WEDA, pasta_WEDA, t_window)
    
    print('Feature extraction completed')
    print("WEDA Fall Dataset Done")
    print("__________________________________________________")


if __name__ == "__main__":
    main()



