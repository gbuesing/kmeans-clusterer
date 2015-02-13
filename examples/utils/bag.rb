require 'stopwords'
require 'fast_stemmer'
require 'narray'


class BagOfWords
  attr_reader :term_index, :doc_hashes, :doc_count, :doc_frequency

  DEFAULT_OPTS = {
    # tf: :raw,
    idf: false,
    stopwords: :en,
    stem: true,
    min_term_length: 3
  }

  def initialize opts = {}
    @opts = DEFAULT_OPTS.merge opts
    @index = 0
    @doc_count = 0
    @term_index = Hash.new do |hsh, key|
      val = hsh[key] = @index
      @index += 1
      val
    end
    @doc_frequency = Hash.new(0)
    @stopwords = Stopwords::Snowball::Filter.new(@opts[:stopwords]) if @opts[:stopwords]
    @doc_hashes = []
  end

  def terms_count
    @index
  end

  def add_docs docs
    docs.each {|doc| add_doc(doc) }
  end

  def add_doc doc
    @doc_count += 1
    terms = extract_terms doc
    doc_hash = create_raw_doc_hash terms
    normalize_tf doc_hash
    # update_doc_hash_tf(doc_hash) unless @opts[:tf] == :raw
    update_doc_frequency doc_hash
    @doc_hashes << doc_hash
  end

  def to_a
    apply_idf_weighting! if @opts[:idf] && !@idf_weighting_applied

    @doc_hashes.map do |doc_hash|
      vec = Array.new(@index, 0)

      doc_hash.each do |k, v|
        vec[k] = v
      end

      vec
    end
  end

  def to_matrix float_precision = :double
    apply_idf_weighting! if @opts[:idf] && !@idf_weighting_applied

    typecode = { double: NArray::DFLOAT, single: NArray::SFLOAT }[float_precision]
    matrix = NMatrix.new(typecode, terms_count, doc_count)

    @doc_hashes.each_with_index do |doc_hash, i|
      doc_hash.each do |k, v|
        matrix[k, i] = v
      end
    end

    matrix
  end

  private

    def apply_idf_weighting!
      @doc_hashes.each do |doc_hash|
        doc_hash.each do |k, v|
          idf = calculate_idf k
          doc_hash[k] = v * idf
        end
      end

      @idf_weighting_applied = true
    end

    def extract_terms doc
      terms = doc.downcase.gsub(/(\d|\s|\W)+/, ' ').strip.split(/\s/)
      terms.reject! {|t| t.length < @opts[:min_term_length]} if @opts[:min_term_length]
      terms = @stopwords.filter(terms) if @stopwords
      terms.map! {|t| t.stem} if @opts[:stem]
      terms
    end

    def create_raw_doc_hash terms
      out = terms.inject(Hash.new(0)) do |hsh, term| 
        index = @term_index[term]
        hsh[index] +=1
        hsh
      end
    end

    def normalize_tf doc_hash
      norm = Math.sqrt doc_hash.values.map {|v| v**2}.reduce(:+)
      doc_hash.each do |k, v|
        doc_hash[k] = v / norm
      end
    end

    # def update_doc_hash_tf doc_hash
    #   max = doc_hash.values.max if @opts[:tf] == :augmented

    #   doc_hash.each do |k, v|
    #     doc_hash[k] = calculate_tf(v, max)
    #   end
    # end

    # def calculate_tf tf, max
    #   case @opts[:tf]
    #   when :binary
    #     1
    #   when :log
    #     1 + Math.log(tf)
    #   when :augmented
    #     0.5 + (0.5 * tf) / max.to_f
    #   end 
    # end

    def update_doc_frequency doc_hash
      doc_hash.each_key do |k|
        @doc_frequency[k] += 1
      end
    end

    def calculate_idf term_id
      Math.log (@doc_count / @doc_frequency[term_id].to_f)
    end
end
