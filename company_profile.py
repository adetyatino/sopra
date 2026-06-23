# company_profile.py
# Berkas konfigurasi statis untuk menyimpan basis pengetahuan (Knowledge Base) PT Solusi Prima Packaging

COMPANY_KNOWLEDGE = {
    "qa_bank": {
    "tentang_perusahaan": (
        "**Jawaban AI:**\n\n"
        "**SOPRA (PT Solusi Prima Packaging)** adalah perusahaan manufaktur dan penyedia solusi "
        "kemasan (packaging) terkemuka. Platform kecerdasan bisnis SOPRA ini dirancang untuk "
        "mengonversi data transaksi mentah menjadi strategi taktis real-time bagi para "
        "reseller guna mengoptimalkan rantai pasok serta meningkatkan volume penjualan."
    ),
    "tujuan_sistem": (
        "**Jawaban AI:**\n\n"
        "Tujuan utama analitik sistem SOPRA meliputi:\n"
        "1. Memberikan rekomendasi produk personal (*cross-selling*) yang akurat.\n"
        "2. Membantu reseller merancang skema promosi adaptif berbasis segmen pelanggan.\n"
        "3. Memprediksi volume permintaan barang untuk mencegah kehabisan stok (*stockout*)."
    ),
    "default_response": (
        "**Jawaban AI:**\n\n"
        "Pertanyaan diterima. Sebagai asisten model pintar SOPRA, saya dapat membantu Anda menjawab informasi "
        "dasar perusahaan, strategi promosi segmen, serta parameter Reorder Point inventori gudang."
    ),
    },
    # --- BANK DATA 100 Q&A TERSTRUKTUR (KNOWLEDGE BASE BOT) ---
    "qa_bank": {
        # KATEGORI 1: PROFIL & INFORMASI UMUM PERUSAHAAN (1 - 20)
        "apa itu sopra": "SOPRA adalah singkatan dari PT Solusi Prima Packaging, sebuah perusahaan manufaktur terkemuka yang berfokus pada penyediaan produk kemasan berkualitas tinggi serta solusi kecerdasan bisnis untuk ekosistem distribusi produk.",
        "apa kepanjangan dari nama sopra": "Kepanjangan dari SOPRA adalah Solusi Prima Packaging.",
        "bergerak di bidang apa pt solusi prima packaging": "Perusahaan bergerak di bidang manufaktur kemasan (packaging), memproduksi berbagai macam wadah primer, sekunder, dan komponen pendukung kemasan untuk berbagai industri.",
        "apa produk utama yang dijual oleh sopra": "Produk utamanya meliputi wadah kemasan seperti Botol PET, Botol Kaca Amber, Karton Box Die Cut, Standing Pouch Aluminium, Tray Makanan Mika, serta komponen pelengkap seperti Tutup Botol Ulir dan Stiker Label Vinyl.",
        "di mana fokus kontribusi sistem kecerdasan bisnis sopra": "Sistem fokus pada pemrosesan log transaksi penjualan mentah menjadi wawasan taktis dan prediksi otomatis guna membantu operasional manajemen stok dan efisiensi penjualan para reseller.",
        "siapa target pengguna utama dari platform digital sopra": "Penguji komite evaluasi internal, manajemen distributor pusat, serta para mitra mitra reseller operasional di lapangan.",
        "apa visi utama dari pt solusi prima packaging melalui platform analitik ini": "Mengintegrasikan ekosistem manufaktur kemasan tradisional dengan teknologi analitik berbasis data guna mendorong efisiensi rantai pasok global.",
        "ke mana laporan atau data demonstrasi aplikasi ini harus dikirimkan": "Hasil demonstrasi harus dikirimkan langsung melalui korespondensi email resmi komite evaluasi di alamat: it_pm@solusi-pack.com.",
        "apa email resmi untuk pengiriman output studi kasus sopra": "Alamat email resminya adalah it_pm@solusi-pack.com.",
        "mengapa sopra membangun platform otomasi berbasis data ini": "Untuk meminimalkan risiko hilangnya potensi omset (opportunity loss) akibat masalah ketidakseimbangan stok dan promosi yang tidak tepat sasaran di tingkat reseller.",
        "apa tantangan utama yang dihadapi oleh lini penjualan sopra sebelumnya": "Tingginya fluktuasi permintaan barang di pasar, risiko kehabisan stok fisik (stockout), serta strategi pemasaran promosi yang belum terarah dengan baik.",
        "apa peran teknologi ai dalam ekosistem manufaktur sopra": "Berfungsi sebagai mesin kecerdasan (Intelligence Engine) untuk segmentasi otomatis, pola asosiasi belanja, dan prediksi runtun waktu pasokan gudang.",
        "apakah sopra menjual produk kemasan makanan": "Ya, salah satu varian produknya adalah Tray Makanan Mika 4 Sekat yang diklasifikasikan ke dalam jenis kemasan primer.",
        "apakah pt solusi prima packaging melayani penjualan partai besar": "Ya, model bisnis perusahaan mendukung distribusi inventori massal melalui sistem kemitraan reseller resmi.",
        "apa segmen industri yang menggunakan kemasan dari sopra": "Industri makanan, minuman, kecantikan (skincare), herbal alami, hingga sektor UMKM kuliner.",
        "apa arti nilai margin pada data transaksi sopra": "Nilai margin mencerminkan keuntungan bersih yang diperoleh dari selisih harga jual dengan biaya produksi atau modal dasar per unit produk.",
        "bagaimana sistem sopra menangani pencatatan data yang corrupt": "Menggunakan fungsi pipeline data berbasis imputation code untuk membersihkan nilai kosong atau mengubah tipe data secara otomatis agar tidak memicu error runtime.",
        "apa yang dimaksud dengan kemasan primer pada katalog produk sopra": "Kemasan primer adalah material kemasan yang bersentuhan langsung dengan produk isi, seperti botol minuman atau wadah mika makanan.",
        "apa contoh kemasan sekunder yang diproduksi oleh pt solusi prima packaging": "Contohnya adalah Karton Box Die Cut ukuran 20x15x8 yang berfungsi melindungi kemasan primer di dalamnya.",
        "apa peran tim it pm di pt solusi prima packaging": "Mengawasi jalannya evaluasi komite proyek teknologi informasi, standarisasi aplikasi, dan pengembangan ekosistem platform intelijen bisnis perusahaan.",

        # KATEGORI 2: SISTEM KECERDASAN DASHBOARD & PIPELINE (21 - 40)
        "apa saja tiga file csv utama yang dikelola dalam pipeline data sopra": "Berkas data_transaksi.csv, data_produk.csv, dan data_pelanggan.csv.",
        "mengapa fungsi load_data di dashboard menggunakan decorator @st.cache_data": "Guna mengoptimalkan kecepatan loading aplikasi dengan menyimpan data di memori lokal, sehingga tidak perlu membaca ulang file CSV dari awal setiap kali user berinteraksi.",
        "apa kegunaan fungsi .merge() pada script ingestion data sopra": "Untuk menghubungkan data transaksi dengan tabel produk berdasarkan kunci kesamaan product_id, agar deskripsi nama asli produk dapat muncul secara informatif di grafik visualisasi.",
        "mengapa program melakukan casting pd.to_datetime pada kolom tanggal": "Agar format teks mentah tanggal berubah menjadi objek waktu yang sah, sehingga proses perhitungan analisis deret waktu (time-series) berjalan akurat.",
        "fungsi dari instruksi .str.strip() pada pembersihan data": "Menghilangkan spasi kosong yang tidak sengaja terinput di awal atau akhir teks kode ID agar proses relasi pencocokan antar tabel tidak mengalami kegagalan.",
        "bagaimana cara dashboard mengatasi nilai kuantitas penjualan yang bernilai negatif": "Sistem melakukan proteksi penyaringan (filtering) otomatis dengan hanya meloloskan baris data yang memiliki nilai qty lebih besar atau sama dengan nol.",
        "mengapa library logging diaktifkan di dalam script": "Sebagai standarisasi level Data/AI Engineering untuk mencatat jejak proses pengolahan data serta menangkap letak error secara sistematis di server.",
        "apa parameter yang digunakan untuk menyatukan tabel transaksi dengan tabel produk": "Kolom kunci penghubungnya adalah kode identitas produk unik yaitu product_id.",
        "apa yang terjadi jika file data_transaksi.csv tidak ditemukan di folder": "Fungsi pengecualian (try-except) akan menangkap error tersebut, lalu mengembalikan DataFrame kosong sambil menampilkan pesan peringatan di antarmuka Streamlit.",
        "kolom apa saja yang wajib ada di dalam struktur file data_transaksi.csv": "Kolom order_id, tanggal, reseller_id, customer_id, product_id, qty, harga, dan margin.",
        "kolom apa saja yang menyusun arsitektur berkas data_produk.csv": "Kolom product_id, nama_produk, kategori, harga_satuan, dan stok.",
        "kolom apa saja yang terdapat pada struktur file data_pelanggan.csv": "Kolom customer_id, nama_pelanggan, jumlah_pembelian, terakhir_beli, dan feedback.",
        "apa fungsi dari argumen errors='coerce' pada fungsi pd.to_numeric": "Mengubah paksa data teks yang rusak atau tidak valid menjadi nilai kosong (NaN) agar tidak merusak kalkulasi matematika, yang kemudian diisi nilai default menggunakan fungsi .fillna().",
        "mengapa visualisasi dashboard menggunakan plotly express dibandingkan library grafis standar": "Karena Plotly Express menghasilkan visualisasi grafik interaktif yang dinamis, responsif, dan dapat menampilkan informasi detail saat kursor diarahkan ke elemen grafik.",
        "bagaimana sistem mengantisipasi perbedaan penamaan kolom antara kata 'produk' dengan 'nama_produk'": "Menggunakan fungsi pengecekan kondisi otomatis untuk mengubah nama (rename) kolom secara fleksibel di awal sesi ingestion data.",
        "apa fungsi utama dari tab pertama pada antarmuka pengguna dashboard": "Menyajikan matriks asosiasi keranjang belanja (Market Basket Analysis) serta rekomendasi produk personal per pelanggan.",
        "apakah dashboard ini mendukung pengawasan performa stok secara real-time": "Ya, melalui visualisasi data agregasi volume penjualan dan tabel indikator status pengadaan stok kritis.",
        "mengapa arsitektur dashboard dibagi ke dalam beberapa tab terpisah": "Untuk menyederhanakan alur navigasi pengguna (user experience) sehingga pemegang kebijakan dapat fokus memantau satu topik analisis tanpa terdistraksi informasi lain.",
        "apa library python yang digunakan untuk membangun antarmuka web interaktif aplikasi ini": "Library utama yang digunakan adalah Streamlit.",
        "apa kegunaan parameter use_container_width=True pada objek grafik plotly": "Memastikan ukuran lebar grafik otomatis menyesuaikan diri dengan resolusi layar monitor user secara proporsional.",

        # KATEGORI 3: REKOMENDASI PRODUK & ASOSIASI KERAJANG (41 - 60)
        "bagaimana logika dasar dari fungsi rekomendasi produk personal di sistem sopra": "Sistem menyaring produk populer teratas dari keseluruhan database transaksi yang riwayatnya belum pernah dibeli sama sekali oleh target ID pelanggan tersebut (cross-selling).",
        "apa yang mendasari perhitungan fungsi get_market_basket_association": "Logika Market Basket Analysis sederhana yang melacak nomor identitas pesanan (order_id) yang sama untuk menemukan pasangan item barang lain yang paling sering dibeli secara bersamaan.",
        "apa itu analisis keranjang belanja (market basket analysis) bagi seorang reseller": "Metode penemuan pola untuk mengetahui kecenderungan perilaku konsumen dalam membeli beberapa jenis produk kemasan sekaligus dalam satu sesi transaksi belanja.",
        "apa keuntungan mengetahui produk yang sering dibeli bersamaan bagi operasional gudang": "Reseller dapat menempatkan produk-produk terkait di rak penyimpanan yang berdekatan atau mengemasnya langsung dalam satu paket penjualan bundling siap kirim.",
        "mengapa sistem menampilkan rekomendasi item populer jika riwayat transaksi pembeli masih baru": "Sebagai strategi mengatasi masalah cold start pada sistem rekomendasi, di mana pengguna baru belum memiliki basis data historis aktivitas pembelian yang cukup.",
        "apa kegunaan parameter top_n=3 pada fungsi rekomendasi": "Membatasi jumlah keluaran produk rekomendasi terbaik hanya sebanyak 3 item teratas agar fokus dan tidak membingungkan pengguna.",
        "bagaimana cara memunculkan rekomendasi produk personal di dashboard": "Pilih salah satu kode Customer ID pada menu dropdown di Tab 1, kemudian tekan tombol 'Generate Rekomendasi Personal'.",
        "jika pembeli memilih 'botol pet bening 500ml', apa komponen pelengkap yang idealnya muncul sebagai produk asosiasi": "Tutup Botol Ulir 28mm atau Stiker Label Vinyl Tahan Air, karena fungsionalitas penggunaannya saling melengkapi dalam pengemasan produk akhir.",
        "berdasarkan visualisasi volume penjualan global, apa indikator penentu produk terlaris": "Dilihat dari nilai total penjumlahan kuantitas akumulatif (qty) terbesar yang terjual sepanjang periode pencatatan log transaksi.",
        "apa fungsi visualisasi grafik batang (bar chart) pada tab rekomendasi produk": "Menampilkan peringkat kontribusi volume penjualan dari seluruh lini SKU produk dari yang tertinggi hingga terendah secara transparan.",
        "apa arti istilah sku pada katalog produk pt solusi prima packaging": "SKU adalah singkatan dari Stock Keeping Unit, kode identifikasi unik untuk membedakan jenis spesifikasi barang seperti ukuran, material, dan harga jual produk kemasan.",
        "mengapa data kualitatif masukan seperti feedback pelanggan ikut dibaca": "Untuk mengukur tingkat kepuasan layanan distribusi dan membantu memahami preferensi kepatuhan reseller terhadap pasokan kemasan pusat.",
        "bagaimana cara mendeteksi pola transaksi cross-selling lintas kategori": "Melalui matriks keterikatan transaksi yang menghubungkan jenis kemasan primer (misalnya Standing Pouch) dengan jenis kemasan sekunder (seperti Karton Box).",
        "apa dampak konkret implementasi algoritma rekomendasi bagi omset reseller": "Membantu menaikkan nilai rata-rata transaksi per pesanan (Average Order Value) karena konsumen terdorong untuk membeli barang pelengkap tambahan.",
        "apakah sistem rekomendasi ini menggunakan library berat seperti deep learning": "Tidak, sistem menggunakan pendekatan komputasi berbasis frekuensi kemunculan matriks transaksi terhitung (frequency count-based), sehingga performa eksekusinya sangat instan.",
        "apa yang terjadi jika semua produk di katalog sudah pernah dibeli oleh pembeli": "Sistem akan otomatis mengembalikan daftar produk dengan volume penjualan tertinggi secara umum di pasar sebagai alternatif rekomendasi reguler.",
        "apa yang dimaksud dengan kode produk 'pkg-btl-01' dalam database transaksi": "Itu adalah kode unik SKU untuk produk Botol PET Bening ukuran 500ml.",
        "apa kode produk yang digunakan untuk produk standing pouch aluminium 250g": "Produk tersebut terdaftar dengan kode SKU unik PKG-PCH-01.",
        "bagaimana grafik plotly membedakan volume penjualan tinggi dan rendah pada visualisasi": "Menggunakan gradasi warna kontras otomatis (color scaling) yang berubah sesuai dengan besaran nilai kuantitas item terjual.",
        "mengapa nama produk nyata lebih diutamakan muncul di visualisasi dibandingkan kode id produk": "Agar lembar laporan hasil analisis bersifat komunikatif, mudah dipahami secara intuitif oleh pengguna awam atau mitra reseller tanpa perlu menghafal arti kode SKU.",

        # KATEGORI 4: SEGMENTASI PELANGGAN & MATRIKS RFM (61 - 80)
        "apa kepanjangan dari konsep analisis matriks rfm": "RFM merupakan singkatan dari Recency (Kebaruan Transaksi), Frequency (Kerapangan Transaksi), dan Monetary (Nilai Total Kontribusi Finansial).",
        "apa definisi dari indikator 'recency' dalam algoritma segmentasi sopra": "Jumlah hitungan hari yang telah berlalu sejak tanggal transaksi paling terakhir dilakukan oleh seorang pelanggan hingga batas waktu penutupan database transaksi.",
        "apa yang diukur melalui indikator 'frequency' pada analisis perilaku pelanggan": "Total jumlah transaksi pembelian unik (berdasarkan nomor order_id yang berbeda) yang telah diselesaikan oleh konsumen selama masa aktif kemitraan.",
        "apa parameter dasar yang digunakan untuk menghitung nilai 'monetary' di studi kasus ini": "Total nilai jumlahan kontribusi margin keuntungan keuntungan bersih yang dihasilkan dari seluruh riwayat pembelian akun pelanggan tersebut.",
        "mengapa pembagian skor rfm pada script menggunakan fungsi pd.qcut berbasis kuantil": "Agar pembagian porsi kelompok batas skor adil secara matematis serta terhindar dari error runtime akibat adanya nilai peringkat yang kembar atau duplikat pada database (duplicate rank bins error).",
        "apa saja pengelompokan nama segmen pelanggan resmi sesuai dengan ekspektasi dokumen studi kasus": "Segmen Loyal, segmen At Risk, segmen New Customer, dan segmen Big Spender.",
        "bagaimana ciri karakteristik dari kelompok segmen pelanggan berkategori 'loyal'": "Memiliki frekuensi transaksi yang sangat rapat dan intens, transaksi terbaru tercatat aktif, serta memberikan kontribusi nilai keuntungan margin yang tinggi bagi perusahaan.",
        "apa skema strategi promosi taktis yang direkomendasikan bagi pelanggan jenis 'loyal'": "Memberikan akses perlakuan eksklusif, program loyalitas poin reward, atau merchandise khusus tanpa perlu memotong margin keuntungan melalui pemberian diskon harga secara berlebihan.",
        "bagaimana indikator performa perilaku belanja dari kelompok segmen 'new customer'": "Memiliki nilai kebaruan bertransaksi yang sangat dekat (baru pertama kali membeli), namun frekuensi akumulatif total pembeliannya masih bernilai rendah.",
        "strategi pemasaran promosi apa yang paling efektif untuk memicu repeat-order segmen 'new customer'": "Memberikan welcome voucher potongan belanja langsung yang dibatasi waktu khusus untuk transaksi pembelian kedua mereka.",
        "apa arti perilaku belanja dari kelompok pelanggan yang masuk dalam segmen 'at risk'": "Merupakan tipe akun pelanggan yang di masa lalu memiliki riwayat frekuensi belanja yang tinggi, namun sudah lama absen dan tidak kembali melakukan transaksi pembelian ulang.",
        "skema promo apa yang disarankan untuk melakukan re-engagement terhadap pelanggan 'at risk'": "Mengirimkan penawaran khusus kupon promosi gratis ongkos kirim terbatas waktu atau program bertema khusus untuk menarik kembali minat interaksi mereka.",
        "bagaimana profil belanja dari pelanggan kelompok 'big spender'": "Akun yang memiliki tingkat frekuensi belanja yang jarang, namun sekali melakukan transaksi pembelian langsung memborong barang dalam volume kuantitas besar dengan nilai keuntungan yang masif.",
        "bentuk strategi pelayanan apa yang harus dipersiapkan khusus untuk segmen 'big spender'": "Memberikan skema potongan harga grosir bertingkat serta jaminan prioritas utama pengalokasian ketersediaan stok fisik barang agar tidak berpindah ke produsen pesaing.",
        "apa kegunaan fungsi algoritma kmeans pada modul intelligence engine dashboard": "Sebagai metode clustering machine learning unsupervised pendukung untuk memetakan sebaran pola alami pengelompokan kemiripan karakter konsumen secara objektif.",
        "mengapa data fitur rfm perlu melewati proses transformasi standardscaler sebelum masuk k-means": "Untuk menyamakan skala satuan nilai antar fitur (karena hari, frekuensi kali, dan nilai uang rupiah memiliki rentang nominal yang timpang) agar hasil pembentukan kluster tidak bias.",
        "apa fungsi dari visualisasi grafik tebar (scatter plot) pada tab segmentasi": "Memetakan posisi koordinat tiap pelanggan dalam ruang matriks tiga dimensi (RFM Space) dengan pembeda warna kontras berdasarkan label segmen mereka.",
        "apa arti dari ukuran besar kecilnya bulatan lingkaran pada visualisasi grafik scatter segmen": "Ukuran besar bulatan mencerminkan tingkat intensitas frekuensi kunjungan atau kuantitas transaksi pembelian dari akun pembeli tersebut.",
        "bagaimana cara meredam risiko hilangnya pelanggan potensial (churn) lewat analisis rfm": "Dengan mendeteksi secara dini pergerakan akun-akun loyal yang nilai indikator kebaruannya (Recency) mulai membengkak membesar untuk segera diberikan stimulus promo.",
        "berapa jumlah kluster optimal yang diatur pada parameter inisialisasi model k-means di script": "Jumlah kelompok kluster diatur sebanyak n_clusters=3 dengan parameter penguncian kestabilan acak random_state=42.",

        # KATEGORI 5: PREDIKSI PERMINTAAN STOK & REORDER POINT (81 - 100)
        "apa fungsi utama dari modul prediksi permintaan stok (demand forecasting engine) bagi mitra reseller": "Menghitung estimasi proyeksi jumlah kuantitas kebutuhan produk untuk siklus satu minggu ke depan guna dasar perencanaan anggaran belanja pengadaan barang.",
        "bagaimana cara sistem menentukan produk mana yang dianalisis pada modul prediksi runtun waktu": "Sistem secara cerdas menggunakan fungsi .idxmax() untuk mendeteksi dan mengunci secara otomatis satu nama produk kemasan yang memegang rekor penjualan total unit terbanyak.",
        "mengapa metode moving average atau rata-rata terbobot sederhana dipilih untuk modul forecasting ini": "Karena sifatnya yang sangat ringan secara kebutuhan resource komputasi di sisi perangkat lokal reseller, mudah dipahami secara transparan, serta terbukti tangguh meredam fluktuasi gangguan data harian (noise).",
        "apa yang dimaksud dengan parameter 'lead time' pada manajemen logistik gudang sopra": "Durasi masa tunggu waktu pengiriman yang dibutuhkan sejak tombol pesanan restock dikirimkan dari reseller hingga barang fisik mendarat resmi di pintu gudang penyimpanan.",
        "berapa estimasi ketetapan waktu 'lead time' yang disimulasikan dalam logika perhitungan pasokan": "Estimasi waktu tunggu pengiriman barang disimulasikan selama 3 hari kerja perjalanan logistik.",
        "apa itu 'safety stock' dalam dunia manajemen rantai pasok (supply chain)": "Batas jumlah persediaan stok barang cadangan minimum yang sengaja disimpan untuk mengamankan toko dari risiko lonjakan permintaan pasar yang mendadak atau keterlambatan pengiriman kurir pusat.",
        "bagaimana rumus matematis untuk mencari nilai safety stock yang digunakan dalam program": "Menggunakan standar tingkat pelayanan pasokan (Service Level Z-Score) sebesar 95 persen yang dikonversikan menjadi pengali nilai konstan 1.65, dikalikan nilai standar deviasi dari data histori kuantitas penjualan harian aktual produk.",
        "apa itu 'reorder point' (rop) atau alarm titik pemesanan kembali": "Indikator batas jumlah kuantitas stok sisa di rak penyimpanan yang menjadi alarm peringatan mutlak bagi reseller untuk segera melakukan order pengisian barang ulang ke distributor pusat.",
        "apa formula rumus matematika di balik perhitungan nilai reorder point (rop)": "Nilai ROP dihitung dari hasil formula: (Rata-rata Kuantitas Permintaan Harian x Durasi Masa Tunggu Lead Time) + Nilai Batas Safety Stock.",
        "apa dampak fatal yang terjadi jika reseller mengabaikan sinyal alarm batas kritis reorder point": "Berisiko tinggi mengalami situasi kekosongan barang di rak (stockout) saat konsumen datang membeli, yang berujung pada hilangnya potensi omset keuntungan operasional toko.",
        "apa kegunaan visualisasi grafik garis (line chart) pada tab prediksi permintaan": "Menampilkan pola fluktuasi naik turunnya volume penjualan produk dari hari ke hari serta posisi letak garis batas kritis pertahanan inventori gudang.",
        "bagaimana cara membaca garis batas merah horizontal putus-putus pada grafik tren peramalan stok": "Jika grafik garis penjualan harian aktual berada mendekati atau berada di bawah posisi garis horizontal merah tersebut, artinya gudang dalam kondisi darurat pengadaan pasokan ulang.",
        "mengapa kode fungsi perhitungan matematika rop sengaja ditampilkan di dalam komponen expander dashboard": "Demi pemenuhan transparansi pengujian di hadapan komite evaluasi penilai, membuktikan validitas penulisan algoritma logika rekayasa data (engineering logic).",
        "kondisi apa yang menyebabkan modul prediksi menampilkan pesan peringatan data riwayat transaksi belum mencukupi": "Jika pencatatan deret waktu tanggal transaksi unik yang merekam aktivitas penjualan item produk pilihan tersebut berjumlah kurang dari batas minimum pengamatan sampel deret.",
        "berapa target prediksi kuantitas jangka waktu ke depan yang dihasilkan sistem": "Target estimasi dirancang langsung untuk memproyeksikan kebutuhan total unit barang untuk jangka waktu 7 hari ke depan (minggu depan).",
        "apa kaitan antara nilai standar deviasi penjualan dengan penentuan tingkat keamanan stok": "Semakin tinggi nilai standar deviasi (artinya penjualan harian sangat tidak stabil dan fluktuatif), maka nilai Safety Stock yang direkomendasikan otomatis akan membesar untuk perlindungan ekstra.",
        "apakah model peramalan ini otomatis memperbarui hasil hitungannya saat file csv diperbarui": "Ya, karena sifat fungsi pipeline data yang dinamis, setiap kali baris data transaksi baru ditambahkan ke dalam file CSV, nilai rata-rata dan ROP akan otomatis menyesuaikan diri secara real-time.",
        "bagaimana cara memperkecil nilai pemborosan biaya penyimpanan gudang (holding cost) tanpa memicu kehabisan stok": "With menerapkan presisi parameter kalkulasi ROP yang pas, sehingga barang pasokan baru dari pusat tiba tepat di saat jumlah barang cadangan di rak hampir habis.",
        "siapa yang bertanggung jawab menanggung pemeliharaan jumlah stok minimum di lapangan": "Masing-masing mitra reseller mandiri yang mengelola operasional wilayah distribusi penjualan eceran di areanya.",
        "apa tujuan akhir pengisian lembar rangkuman eksekutif taktis satu halaman pada aplikasi": "Menyediakan instrumen panduan aksi korektif operasional yang cepat dan komprehensif bagi pelaku bisnis tanpa harus membaca baris baris baris kode program mentah.",

        # KATEGORI 6: STRATEGIS & KEMITRAAN BISNIS (101 - 110)
        "siapa saja target mitra bisnis utama sopra": "SOPRA menargetkan reseller, distributor, UMKM, toko retail, serta perusahaan yang membutuhkan solusi kemasan untuk mendukung aktivitas penjualan dan distribusi produk mereka.",
        "bagaimana sopra membantu reseller meningkatkan penjualan": "SOPRA menyediakan berbagai pilihan produk kemasan dengan kualitas yang konsisten, harga kompetitif, dan ketersediaan stok yang memadai. Selain itu, reseller dapat memanfaatkan analisis data penjualan untuk menentukan produk yang paling diminati pelanggan.",
        "produk apa yang dapat direkomendasikan kepada pelanggan yang membeli kemasan makanan": "Pelanggan yang membeli kemasan makanan dapat direkomendasikan produk pelengkap seperti plastik seal, stiker label, paper bag, sendok garpu sekali pakai, dan bubble wrap untuk meningkatkan nilai transaksi.",
        "bagaimana sopra membantu reseller menghindari kehabisan stok": "SOPRA dapat memanfaatkan data historis penjualan untuk memprediksi permintaan produk dan menghitung reorder point sehingga reseller dapat melakukan pemesanan ulang sebelum stok habis.",
        "bagaimana sopra mengidentifikasi pelanggan yang berpotensi menjadi pelanggan loyal": "Dengan analisis RFM (Recency, Frequency, Monetary), SOPRA dapat mengelompokkan pelanggan berdasarkan frekuensi pembelian, nilai transaksi, dan waktu pembelian terakhir untuk menentukan pelanggan loyal dan pelanggan potensial.",
        "bagaimana ai dapat membantu mitra bisnis sopra dalam mengambil keputusan": "AI dapat menganalisis pola transaksi, memberikan rekomendasi produk yang relevan, memprediksi kebutuhan stok, serta membantu menyusun strategi promosi yang lebih tepat sasaran berdasarkan perilaku pelanggan.",
        "apa keunggulan sopra dibandingkan pemasok kemasan lainnya bagi reseller": "SOPRA menawarkan variasi produk yang lengkap, dukungan distribusi yang baik, kualitas produk yang konsisten, serta peluang pemanfaatan data dan AI untuk meningkatkan efektivitas penjualan dan pengelolaan stok.",
        "saya adalah reseller baru produk apa yang sebaiknya saya stok terlebih dahulu": "Prioritaskan produk dengan tingkat penjualan tertinggi dan perputaran cepat seperti kemasan makanan, plastik pengiriman, dan paper bag. Data transaksi historis dapat digunakan untuk menentukan produk yang paling diminati di wilayah Anda.",
        "mitra bisnis mana yang memiliki potensi pertumbuhan penjualan tertinggi": "Mitra yang berada pada segmen Potential Loyalist memiliki frekuensi pembelian yang cukup tinggi namun nilai transaksi masih dapat ditingkatkan melalui program bundling, diskon volume, atau rekomendasi produk pelengkap.",
        "strategi apa yang sebaiknya dilakukan sopra untuk meningkatkan penjualan melalui mitra bisnis": "SOPRA dapat menerapkan rekomendasi produk berbasis AI, segmentasi pelanggan untuk promosi yang lebih personal, prediksi stok untuk mengurangi kehabisan barang, serta program loyalitas untuk mempertahankan reseller dengan performa terbaik."
    }
}
    
