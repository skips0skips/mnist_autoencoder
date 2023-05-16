import matplotlib.pyplot as plt

class Plt_show:
    def plt_show(reconstruction, epoch, avg_loss, n_epochs,X_val):
        '''
        '''
        fig, axs = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
        for i in range(10):
            axs[0][i].imshow(X_val[0][i].permute(1, 2, 0).cpu(), cmap='gist_gray')
            axs[0][i].axis('off')
            axs[0][i].set_title('X {}'.format(i))
            axs[1][i].imshow(reconstruction[i].permute(1, 2, 0).cpu(), cmap='gist_gray')
            axs[1][i].axis('off')
            axs[1][i].set_title('reconstructed {}'.format(i))
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, n_epochs, avg_loss))
        plt.show()

    def image_show(image):
        plt.figure(figsize=(50, 50))
        for i, img in enumerate(image):
            plt.subplot(1, 15, i+1)
            plt.imshow(img.permute(1, 2, 0).cpu().detach().numpy(), cmap='gist_gray')