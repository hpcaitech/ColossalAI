def get_data_per_category(data, categories):
    data_per_category = {category: [] for category in categories}
    for item in data:
        category = item["category"]
        if category in categories:
            data_per_category[category].append(item)

    return data_per_category
