def preprocess(names_df):
    # Step 1: Lowercase
    names_df["name"] = names_df["name"].str.lower()

    # Step 2: Split individual characters
    names_df["name"] = [list(name) for name in names_df["name"]]

    # Step 3: Pad names with spaces to make all names same length
    name_length = 50
    names_df["name"] = [
        (name + [" "] * name_length)[:name_length] for name in names_df["name"]
    ]

    # Step 4: Encode Characters to Numbers
    names_df["name"] = [
        [max(0.0, ord(char) - 96.0) for char in name] for name in names_df["name"]
    ]

    return names_df
