require 'stopwords'
require 'fast_stemmer'

class Bag
  attr_reader :term_index, :doc_hashes, :doc_count, :doc_frequency

  def initialize
    @index = 0
    @doc_count = 0
    @term_index = Hash.new do |hsh, key|
      val = hsh[key] = @index
      @index += 1
      val
    end
    @doc_frequency = Hash.new(0)
    @stopwords = Stopwords::Snowball::Filter.new "en"
    @doc_hashes = []
  end

  def terms_count
    @index
  end

  def << doc
    @doc_count += 1
    terms = extract_terms doc
    doc_hash = create_doc_hash terms
    update_doc_frequency doc_hash
    @doc_hashes << doc_hash
  end

  def to_a
    @doc_hashes.map do |doc_hash|
      vec = Array.new(@index, 0)
      doc_hash.each do |k, v|
        vec[k] = v
      end
      vec
    end
  end

  def binary!
    @doc_hashes.each do |doc_hash|
      doc_hash.each_key do |k|
        doc_hash[k] = 1
      end
    end
  end

  def tf_idf!
    @doc_hashes.each do |doc_hash|
      max_tf = doc_hash.values.max

      doc_hash.each do |k, v|
        tf = 0.5 + (0.5 * v) / max_tf.to_f
        idf = Math.log (@doc_count / @doc_frequency[k].to_f)
        tf_idf = tf * idf
        doc_hash[k] = tf_idf
      end
    end
  end

  private

    def extract_terms doc
      terms = doc.downcase.gsub(/(\d|\s|\W)+/, ' ').strip.split(/\s/)
      terms.reject! {|t| t.length < 3}
      terms = @stopwords.filter terms
      terms.map! {|t| t.stem}
      terms
    end

    def create_doc_hash terms
      out = terms.inject(Hash.new(0)) do |hsh, term| 
        index = @term_index[term]
        hsh[index] +=1
        hsh
      end
    end

    def update_doc_frequency doc_hash
      doc_hash.each_key do |k|
        @doc_frequency[k] += 1
      end
    end
end
