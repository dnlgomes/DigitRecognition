import csv
import numpy as np
final_data = []
final_data_horizontal = []
final_data_vertical= []
with open('no_label_train.csv', 'rb') as csvfile:
    trava = 0
    file = csv.reader(csvfile)
    for row in file:
        if trava > 0:
            linha = np.array(row).reshape(28,28).astype(int)
            ##codigo das projecoes

            #codigo da projecao horizontal
            sublista_proj_horizontal = []
            for linha_matriz in linha:
                sublista_proj_horizontal.append( np.sum(linha_matriz) )
            final_data_horizontal.append(sublista_proj_horizontal)

            #codigo da projecao vertical
            sublista_proj_vertical = []
            for linha_matriz in np.transpose(linha):
                sublista_proj_vertical.append( np.sum(linha_matriz) )
            final_data_vertical.append(sublista_proj_vertical)
            print(trava)
            ##fim
            # primeiro, segundo, terceiro, quarto = linha[:14, :14], linha[14:, :14], linha[:14, 14:], linha[14:, 14:]
            # quadrantes = [primeiro, segundo, terceiro, quarto]
            # sublista = []
            # for quadrante in quadrantes:
            #     print quadrante
            #     print "-=-=-=-"
            #     not_white = 0.0
            #     for item in quadrante:
            #         not_white = not_white + ( len(item) - list(item).count(0) )
            #         #print not_white
            #     sublista.append(not_white/quadrante.size)
            # #sublista = [np.mean(primeiro), np.mean(segundo), np.mean(terceiro), np.mean(quarto)]
            # final_data.append(sublista)
        trava += 1


#with open('porcentagem_nao_branco2.csv', 'wb') as csvfile:
#    escritor = csv.writer(csvfile)
#    for row in final_data:
#        escritor.writerow(row)

def escrever_csv(nome_arquivo, dado):
    with open(nome_arquivo, 'wb') as csvfile:
        escritor = csv.writer(csvfile)
        for row in dado:
            escritor.writerow(row)

#escrever_csv("porcentagem_nao_branco2.csv", final_data)
escrever_csv("proj_horizontal.csv", final_data_horizontal)
escrever_csv("proj_vertical.csv", final_data_vertical)



