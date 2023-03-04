import pandas as pd


def resume_analysis(im_dict:dict):
    for key, item in dict.items():
        for key2, val in item.items():
            print(key, key2, val)

    df = pd.DataFrame(im_dict)
    print(df.transpose())


if __name__ == "__main__":
    dict = {
        "id_1": {
            "im": 0,
            "mse": 1,
        },
        "id_2": {
            "im": 0,
            "mse": 1,
        }
    }
    resume_analysis(dict)