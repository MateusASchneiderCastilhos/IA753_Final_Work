# Guia de Continuidade: Projeto de BCI (Cérebro Saudável vs. Lesão Medular)

Bem-vindo(a) ao projeto de decodificação de intenção motora via Eletroencefalografia (EEG). Este documento serve como ponto de partida para compreender a arquitetura dos dados processados e os próximos passos lógicos da pesquisa em neuroengenharia.

---

## 1. O Estado Atual dos Dados (O que já foi feito)

O trabalho árduo de higienização de sinais biológicos já foi concluído. Não é necessário lidar com os arquivos brutos `.gdf` originais, a menos que o objetivo seja alterar os parâmetros fundamentais dos filtros. 

Os dados estão consolidados em arquivos Mestre no formato nativo da biblioteca MNE-Python (`.fif`). Estes arquivos contêm as fatias de tempo (`Epochs`) de dezenas de tentativas de movimento, processadas sob um pipeline rigoroso de padrão-ouro:

* **Filtragem Espectral:** Passa-banda de 0.3 Hz a 40 Hz (preservando o MRCP no domínio do tempo e as bandas Mu, Beta e Gama Baixa no domínio da frequência).
* **Limpeza de Artefatos (ICA + MAD):** Ruídos musculares e piscadas de olhos (EOG) foram isolados e removidos via Análise de Componentes Independentes (Infomax). O treinamento da ICA foi blindado contra transientes extremos utilizando um limiar de 10.4x o Desvio Absoluto da Mediana (MAD).
* **Segmentação (Epoching):** O sinal contínuo foi cortado em janelas de **-1.0s a +3.0s** em relação ao instante do gatilho visual (t=0).
* **Rejeição Estrita:** Qualquer época que ultrapassasse ± 100 µV foi sumariamente descartada para garantir que o sinal provém puramente do córtex.

### Os Arquivos Mestre Disponíveis
* `S02_ME_Mestre_Limpo-epo.fif`: Sujeito Saudável (Execução Motora)
* `S02_MI_Mestre_Limpo-epo.fif`: Sujeito Saudável (Imaginação Motora)
* `P02_SCI_Mestre_Limpo-epo.fif`: Paciente com Lesão Medular Severa (Tentativa de Execução)

> **Nota:** As classes de movimento comuns e diretamente comparáveis entre todos os arquivos são: `'Hand_Open'`, `'Supination'` e `'Pronation'`.

---

## 2. Como Carregar e Manipular os Dados (Quickstart)

Para carregar os dados para a RAM e iniciar as análises, utilize a biblioteca `mne`. Abaixo está o bloco de código fundamental para uso diário:

```python
import mne

# 1. Carregar o arquivo Mestre
caminho = "Data/S02_ME_Mestre_Limpo-epo.fif"
epochs = mne.read_epochs(caminho, preload=True)

# 2. Carregar a topografia padrão (Touca 10-05)
montage = mne.channels.make_standard_montage('standard_1005')
epochs.set_montage(montage, match_case=False, on_missing='ignore')

# 3. Correção de Escala (O "Bug" da Escala do MNE)
# Os dados originais já estavam em microvolts. Para evitar que o MNE
# os multiplique incorretamente ao plotar gráficos, devolvemos a escala a Volts:
epochs.apply_function(lambda x: x / 1e6)

# 4. Correção de Linha de Base (Baseline Correction)
# Zera a voltagem média no segundo anterior ao estímulo para alinhar os gráficos
epochs.apply_baseline(baseline=(-1.0, 0.0))

# 5. Isolando Classes e Canais Específicos
canais_motor = ["C3", "C1", "Cz", "C2", "C4"]
dados_abrir_mao = epochs['Hand_Open'].copy().pick(canais_motor)

# Gerar gráfico do Potencial Relacionado ao Movimento (MRCP)
dados_abrir_mao.average().plot(spatial_colors=True)
