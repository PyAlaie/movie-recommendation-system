import sys, config
from termcolor import colored

args = sys.argv[1:]

def preprocess(*args):
    args = args[0]
    from src.data_preprocess import preprocess

    steps = preprocess.steps

    if len(args) == 0:
        for name, func in steps.items():
            print(colored(name, 'green'))
            func()
    
    elif args[0] == 'help':
        print("Commands:")
        [print('  ',i) for i in steps.keys()]
    
    else:
        command = steps.get(args[0])
        if command:
            command()
        
        else:
            print("Not found!")

def join(*args):
    args = args[0]
    from src.data_preprocess import join

    steps = join.steps

    if len(args) == 0:
        for name, func in steps.items():
            print(name)
            func()
    
    elif args[0] == 'help':
        print("Commands:")
        [print('  ',i) for i in steps.keys()]
    
    else:
        command = steps.get(args[0])
        if command:
            command()
        
        else:
            print("Not found!")

def add_statistics(*args):
    args = args[0]
    from src.data_preprocess import add_statistics

    steps = add_statistics.steps

    if len(args) == 0:
        for name, func in steps.items():
            print(name)
            func()
    
    elif args[0] == 'help':
        print("Commands:")
        [print('  ',i) for i in steps.keys()]
    
    else:
        command = steps.get(args[0])
        if command:
            command()
        
        else:
            print("Not found!")

def fit_models(*args):
    args = args[0]
    from src.recommenders import collaborative_filtering, content_based 

    print("Building CollabrativeFileteringKNN")
    knn = collaborative_filtering.CollabrativeFileteringKNN()
    knn.fit()
    knn.save()

    print("Building CollabrativeFilteringMF")
    mf = collaborative_filtering.CollabrativeFilteringMF()
    mf.fit()
    mf.save()

    print("Building ContentBasedRecommender")
    cb = content_based.ContentBasedRecommender()
    cb.build()
    cb.save()

def evaluate(*args):
    args = args[0]
    from src.recommenders import eval

    eval.main()

def build(*args):
    print(colored('preprocess', 'yellow'))
    preprocess([])

    print(colored('join', 'yellow'))
    join([])

    print(colored('add statistics', 'yellow'))
    add_statistics([])

    print(colored('fit models', 'yellow'))
    fit_models([])

options = {
    "preprocess": preprocess,
    "join": join,
    "add_statistics": add_statistics,
    "fit_models": fit_models,
    "evaluate": evaluate,
    "build": build,
}

if len(args) == 0:
    commands = options.keys()
    print("Commands:")
    [print('  ',i) for i in commands]

else:
    command = options.get(args[0])

    if not command:
        print(f"Command '{args[0]}' not found!")
    
    else:
        command(args[1:])