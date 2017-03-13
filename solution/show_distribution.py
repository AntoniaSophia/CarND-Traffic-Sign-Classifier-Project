for i in range(n_groups):
    ftrain.append(y_train_pp.count(i))
    ftest.append(y_test_pp.count(i))
    fvalid.append(y_valid_pp.count(i))

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.3

opacity = 0.6

rects1 = plt.bar(index, ftrain, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Train images')

rects2 = plt.bar(index + bar_width, ftest, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Test images')

rects2 = plt.bar(index + 2*bar_width, fvalid, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Valid images')

plt.xlabel('Sign ID')
plt.ylabel('Count')
plt.title('Distribution of Dataset')
plt.xticks(index + (2*bar_width) / 2, range(n_groups))
plt.legend()
plt.tight_layout()
plt.show()
