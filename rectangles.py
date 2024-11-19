'''Funções para se trabalhor com retãngulos.'''

def filter_recs_by_size(r, min_w, max_w, min_h, max_h):
    '''Filtra retângulos muito pequenos ou muito grandes.'''
    return min_w < r[2] < max_w and min_h < r[3] < max_h

def union(a, b):
    '''Une dois retângulos.'''
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def inside(a, b):
    '''Retorna se o retângulo 'b' está dentro da área de 'a'. Assumindo que a
    coordenada x de 'a' é menor que 'b'.'''
    return b[1] >= a[1] and b[1]+b[3] <= a[1]+a[3] and b[0]+b[2] <= a[0]+a[2]

def intersec(a, b):
    '''Retorna a razão entre a área de interseção e a soma da área de dois
    retângulos. Assumindo que a coordenada x de 'a' é menor que 'b'. Retorna 0 se
    os retângulos não se sobrepõem.'''
    ay = a[1]+a[3]
    by = b[1]+b[3]
    overy = ay-b[1] if by >= ay else by-a[1]
    area = ((a[0]+a[2])*ay) + ((b[0]+b[2])*by)
    return max(a[0]+a[2]-b[0], 0)*max(overy, 0)/area

def merge_recs(contours, dist_y=15, dist_x=7, overlap=0.25, by_overlap=False):
    '''Junta todos os contornos muito próximos no eixo x ou que estão
    completamente dentro de um maior.'''
    # Ordena os contornos pelo eixo x.
    contours.sort(key=lambda tup: tup[0])
    rectangles_merged = contours
    merge = True
    if by_overlap:
        # Faz todas as uniões possíveis entre os contornos, usando a área de
        # sobreposição.
        while merge:
            merge = False
            prev = rectangles_merged
            rectangles_merged = []
            for i in range(len(prev)-1):
                j = i+1
                while j < len(prev):
                    if intersec(prev[i], prev[j]) > overlap:
                        rectangles_merged.append(union(prev[i], prev[j]))
                        merge = True
                        break
                    j += 1
                if merge:
                    for k in range(i+1, len(prev)):
                        if k == j:
                            continue
                        rectangles_merged.append(prev[k])
                    break
                rectangles_merged.append(prev[i])
                if i == len(prev)-2:
                    rectangles_merged.append(prev[i+1])
    else:
        # Une os retângulos completamente sobrepostos ou considerados próximos.
        while merge:
            merge = False
            prev = rectangles_merged
            rectangles_merged = []
            for i in range(len(prev)-1):
                j = i+1
                while j < len(prev):
                    if (abs(prev[i][1]-prev[j][1]) <= dist_y and prev[j][0]-(prev[i][0]+prev[i][2]) <= dist_x) \
                        or inside(prev[i], prev[j]):
                        rectangles_merged.append(union(prev[i], prev[j]))
                        merge = True
                        break
                    j += 1
                if merge:
                    for k in range(i+1, len(prev)):
                        if k == j:
                            continue
                        rectangles_merged.append(prev[k])
                    break
                rectangles_merged.append(prev[i])
                if i == len(prev)-2:
                    rectangles_merged.append(prev[i+1])
    if len(rectangles_merged) == 0:
        rectangles_merged = prev
    return rectangles_merged

def sort_recs(a, b):
    '''Ordena os retângulos de cima para baixo e da esquerda para a direita.'''
    if abs(a[1]-b[1]) <= 25:
        return a[0] - b[0]
    return a[1] - b[1]
