def tensor_summary(name, t):
    print('%s :: %s, mean: %f, sum: %f, std: %f' % (name, t.shape, t.mean(), t.sum(), t.std()))
