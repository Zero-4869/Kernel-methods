from kernels import Walk, Path, GeoPath, Subtree, WLsubtree, SimpleWLsubtree
from classifier import KernelSVC
# from sklearn.metrics import roc_auc_score
import pickle
import os

with open("./data/training_data.pkl", "rb") as f:
    G_total = pickle.load(f)
    f.close()
with open("./data/training_labels.pkl", "rb") as f:
    y_total = pickle.load(f)
    f.close()
with open("./data/test_data.pkl", "rb") as f:
    G_test = pickle.load(f)
    f.close()

output_file = "./eval_result"
if not os.path.exists(output_file):
    os.mkdir(output_file)

### Kernel
walk = Walk(n=2)
path = GeoPath(n=2)
subtree = Subtree(n=1)
simplewlsubtree = SimpleWLsubtree()
wlsubtree = WLsubtree(ite=4)

### Train
kernel = wlsubtree.kernel
C=50
model = KernelSVC(kernel=kernel, C=C)
model.fit(G_total[:4000], y_total[:4000])

### Validation
# y_val_pred = model.predict(G_val)
# auc_wk = roc_auc_score(y_val, y_val_pred)
# print("val AUC random walk:", auc_wk)

### Test
y_test_pred = model.predict(G_test)

### Write prediction
with open(os.path.join(output_file, "test_pred.csv"), "w") as f:
    f.write("Id,Predicted\n")
    for i in range(len(G_test)):
        f.write("%d,%f\n" % (i+1, y_test_pred[i]))
    f.close()

