/** Berikut adalah contoh sederhana implementasi perceptron dengan JavaScript. 
 * Perceptron akan melakukan klasifikasi biner berdasarkan input dan output. 
 * Perceptron terdiri dari beberapa tahap seperti inisialisasi bobot, forward propagation, 
 * pembaruan bobot berdasarkan error, dan iterasi pelatihan. */

// Kelas Perceptron untuk model jaringan saraf sederhana
class Perceptron {
  // Konstruktor untuk inisialisasi perceptron
  constructor(learningRate = 0.01, epochs = 100) {
    this.learningRate = learningRate; // Tingkat pembelajaran
    this.epochs = epochs; // Jumlah iterasi pelatihan
    this.weights = []; // Bobot model
    this.bias = 0; // Bias model
  }

  // Fungsi aktivasi yang mengembalikan 1 jika sum lebih besar dari 0, jika tidak 0
  activation(sum) {
    return sum > 0 ? 1 : 0;
  }

  // Metode untuk melatih perceptron
  train(inputs, labels) {
    // Menentukan jumlah fitur dari input
    const numFeatures = inputs[0].length;
    
    // Inisialisasi bobot dengan nilai nol
    this.weights = Array(numFeatures).fill(0);
    this.bias = 0; // Inisialisasi bias dengan nol

    // Melakukan iterasi sebanyak epoch
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Iterasi setiap contoh pelatihan
      for (let i = 0; i < inputs.length; i++) {
        const input = inputs[i]; // Data input
        const label = labels[i]; // Label yang benar untuk data input
        
        // Membuat prediksi berdasarkan input saat ini
        const prediction = this.predict(input);
        
        // Menghitung error antara label yang benar dan prediksi
        const error = label - prediction;

        // Memperbarui bobot dan bias berdasarkan error
        for (let j = 0; j < numFeatures; j++) {
          this.weights[j] += this.learningRate * error * input[j];
        }
        this.bias += this.learningRate * error;
      }
    }
  }

  // Metode untuk memprediksi output berdasarkan input
  predict(input) {
    // Menghitung jumlah dengan bobot dan bias
    let sum = this.bias;
    for (let i = 0; i < this.weights.length; i++) {
      sum += this.weights[i] * input[i];
    }
    // Menggunakan fungsi aktivasi untuk mendapatkan output
    return this.activation(sum);
  }
}

// Contoh penggunaan perceptron

// Data input untuk fungsi logika AND
const inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
];

// Label yang benar untuk fungsi logika AND
const labels = [0, 0, 0, 1];

// Membuat objek perceptron dengan tingkat pembelajaran 0.1 dan 100 epoch
const perceptron = new Perceptron(0.1, 100);

// Melatih perceptron dengan data input dan label
perceptron.train(inputs, labels);

// Menampilkan bobot dan bias yang telah dilatih
console.log('Weights:', perceptron.weights);
console.log('Bias:', perceptron.bias);

// Menguji perceptron dengan data input dan menampilkan prediksi
inputs.forEach(input => {
  console.log(`Input: ${input}, Prediction: ${perceptron.predict(input)}`);
});

/** Penjelasan: 
 * 1. Inputs dan Targets
 *  a. inputs: Setiap input adalah pasangan dua angka biner. Misalnya, [1, 1] dan [0, 1].
    b. targets: Ini adalah hasil yang diharapkan dari input. Misalnya, output yang diharapkan untuk [1, 1] adalah 1, dan untuk [0, 0] adalah 0.

  2. Bobot dan Bias
    a. Bobot diinisialisasi secara acak pada awalnya.
    b. Setiap iterasi (epoch) akan memperbarui bobot berdasarkan error antara output yang diprediksi dan target.

  3. Forward Propagation
    a. Keluaran dihitung sebagai hasil dari input dikalikan dengan bobot, ditambah bias. 
    Jika hasilnya lebih besar dari atau sama dengan 0, outputnya adalah 1, jika tidak, outputnya 0 (fungsi aktivasi step).

  4. Pembaruan Bobot
    a. Bobot diperbarui setiap kali terjadi error dalam prediksi. Pembaruan menggunakan rumus:
      wi←wi+η⋅(ytarget−yprediksi)⋅x i
      di mana 𝜂 adalah learning rate, error adalah perbedaan antara target dan prediksi, dan 𝑥𝑖 adalah input.

  5. Epoch
    a. Setiap epoch adalah satu siklus penuh melalui seluruh dataset. 
    Dalam contoh ini, pelatihan dilakukan selama 10 epoch, tapi jumlah epoch bisa disesuaikan sesuai kebutuhan.

  Output:
  Program ini akan mencetak hasil prediksi, error, serta pembaruan bobot dan bias setelah setiap epoch. 
  Setelah pelatihan selesai, perceptron bisa diuji dengan input baru atau yang sama untuk melihat hasil prediksinya.
  Ini adalah contoh sederhana dari perceptron yang melakukan klasifikasi biner pada data input dua dimensi.
 */